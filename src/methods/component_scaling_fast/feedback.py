"""Reuse within a minibatch only; preserve every feedback measurement.

Each boundary dictionary contains M and a suffix of the SAME block components.
Neither M nor a component depends on the boundary. Therefore raw components
can be computed once per sample chunk. Norms and Gram matrices are recomputed
at each original shape: CUDA reductions/GEMMs round differently for 15 modes
versus 43 modes. Algebraic submatrix reuse alone is insufficient for parity.
Ridge scaling, fitting, validation and error contraction remain boundary local.
No cache survives a parameter update. No probe, class or direction is dropped.
"""
import math

import torch

from methods.component_postln.feedback import ComponentFeedback
from methods.component_postln.operators import block_component_operators
from methods.component_scaling.feedback import slice_caches, chunked_vector_responses
from methods.transformer_ebd.component_route_operator import (
    identity_class_operator, sample_output_codes, operator_response,
    select_operator, batched_vector_responses,
)
from methods.transformer_ebd.forward_response_credit import sample_probes


@torch.no_grad()
def fit_with_gram(suffix, tokens, carrier, dictionary, codes, gram, *,
                  epsilon, ridge, trust_ratio, query_chunk):
    """Original streamed trace fit, supplied with the already computed Gram."""
    rhs = []
    for component in dictionary.unbind(1):
        total = carrier.new_zeros(carrier.shape[0])
        for code in codes.split(query_chunk, dim=1):
            raw = torch.einsum("bkc,bcnd->bknd", code, component)
            scale = raw.flatten(2).square().mean(-1).sqrt().clamp_min(1e-20)
            probes = raw / scale[:, :, None, None]
            response = batched_vector_responses(suffix, tokens, probes, epsilon)
            residual = (response - operator_response(carrier, probes)) * scale[:, :, None]
            total += (residual * code).sum(-1).sum(-1)
        rhs.append(total / codes.shape[1])
    rhs = torch.stack(rhs, dim=1)
    # The submatrix has its own ridge normalization, just as in the baseline.
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram + ridge * scale[:, None, None] * identity,
                                      rhs.unsqueeze(-1)).squeeze(-1)
    correction = torch.einsum("br,brcnd->bcnd", coefficients, dictionary)
    limit = trust_ratio * carrier.flatten(1).norm(dim=-1)
    shrink = (limit / correction.flatten(1).norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
    return carrier + correction * shrink[:, None, None, None], coefficients * shrink[:, None], rhs


def cls_suffix(model, tokens, boundary):
    """Optional dead-output elimination in the last post-LN block.

The last block's non-CLS outputs cannot affect the classifier. All keys and
values remain; only unused query rows and tokenwise FFNs are omitted. This is
algebraically identical but smaller GEMMs may round differently in float32.
It is experimental and OFF by default until the numerical audit is passed.
"""
    for block in model.blocks[boundary + 1:-1]:
        tokens, _ = block(tokens)
    block = model.blocks[-1]
    attention = block.attention
    batch, count, width = tokens.shape
    combined = attention.qkv(tokens).view(batch, count, 3, attention.num_heads, attention.head_dim)
    q, k, v = (part.transpose(1, 2) for part in combined.unbind(2))
    scores = (q[:, :, :1] @ k.transpose(-2, -1)) / math.sqrt(attention.head_dim)
    context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, 1, width)
    after_attention = block.ln1(tokens[:, :1] + attention.proj(context))
    branch, _, _ = block.ffn(after_attention)
    final = block.ln2(after_attention + branch)
    return model.head(model.final_norm(final)[:, 0])


class CachedComponentFeedback(ComponentFeedback):
    def __init__(self, model, config, *, cls_only=False):
        super().__init__(model, config)
        self.cls_only = cls_only
        if model.norm_layout != "post":
            raise ValueError("Cached feedback requires the audited post-LN model")

    @torch.no_grad()
    def prepare(self, caches, error):
        cfg, batch, depth = self.config, error.shape[0], len(caches)
        count = cfg.output_probes or self.model.config.num_classes
        chunk = min(cfg.feedback_sample_chunk, batch)
        cycles = cfg.mirror_warmup if self.mirrors.observations == 0 else 1
        for _ in range(cycles):
            self.mirrors.update()
        # Keep the whole-minibatch draws interleaved in legacy boundary order.
        # Reordering execution is safe only AFTER all these draws are complete.
        codes, validation = [], []
        for cache in caches[:-1]:
            reference = error.new_empty(batch, self.model.config.num_classes, 0)
            codes.append(sample_output_codes(reference, count, self.generator))
            validation.append(sample_probes(cache.block_output, cfg.validation_probes,
                                             self.generator, orthogonal=True))
        diagnostics = {"local_mirror_forward_calls": cycles * 8 * (depth - 1),
                       "suffix_forward_calls": 0, "suffix_directional_evaluations": 0,
                       "feedback_sample_chunks": math.ceil(batch / chunk),
                       "feedback_output_probes": count}
        pieces = [[] for _ in caches]
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            subset = slice_caches(caches, start, end)
            carrier = identity_class_operator(self.model, subset)
            if depth > 1:
                families = [carrier[:, None]]
                for index in range(1, depth):
                    parts, _ = block_component_operators(carrier, subset[index], self.model.blocks[index],
                                                         self.mirrors.maps[index])
                    families.append(parts)
                raw = torch.cat(families, dim=1)
                del families, parts
            for boundary in range(depth - 1):
                scale = carrier.flatten(1).norm(dim=-1)[:, None] / raw.flatten(2).norm(dim=-1).clamp_min(1e-20)
                dictionary = raw * scale[:, :, None, None, None]
                gram = dictionary.flatten(2) @ dictionary.flatten(2).transpose(1, 2)
                def suffix(tokens):
                    diagnostics["suffix_forward_calls"] += 1
                    if self.cls_only:
                        return cls_suffix(self.model, tokens, boundary)
                    for block in self.model.blocks[boundary + 1:]:
                        # Drop unused forward caches before the next block.
                        tokens = block(tokens)[0]
                    # LayerNorm is tokenwise; only this row is read by the head.
                    return self.model.head(self.model.final_norm(tokens[:, :1])[:, 0])

                candidate, coefficients, _ = fit_with_gram(
                    suffix, subset[boundary].block_output, carrier, dictionary, codes[boundary][start:end], gram,
                    epsilon=cfg.probe_epsilon, ridge=cfg.operator_ridge,
                    trust_ratio=cfg.trust_ratio, query_chunk=cfg.output_query_chunk)
                probes = validation[boundary][start:end]
                response = chunked_vector_responses(suffix, subset[boundary].block_output, probes,
                                                     cfg.probe_epsilon, cfg.output_query_chunk)
                selected, accepted = select_operator(carrier, candidate, probes, response)
                weight = (end - start) / batch
                values = {
                    "operator_acceptance": float(accepted.float().mean()),
                    "coefficient_mean_square": float(coefficients.square().mean()),
                    "correction_ratio": float(((selected-carrier).flatten(1).norm(dim=-1) /
                                               carrier.flatten(1).norm(dim=-1).clamp_min(1e-20)).mean()),
                }
                for name, value in values.items():
                    key = f"boundary{boundary}_{name}"
                    diagnostics[key] = diagnostics.get(key, 0.0) + weight * value
                diagnostics[f"boundary{boundary}_operator_modes"] = dictionary.shape[1]
                diagnostics["suffix_directional_evaluations"] += weight * 2 * (dictionary.shape[1] * count + cfg.validation_probes)
                pieces[boundary].append(torch.einsum("bc,bcnd->bnd", error[start:end], selected))
                del candidate, coefficients, selected, response, dictionary, gram
                if boundary < depth - 2:
                    # Discard the completed block's components, retaining M.
                    # Release the old large dictionary at once to bound memory.
                    modes = 2 + 3 * self.model.blocks[boundary + 1].attention.num_heads
                    raw = torch.cat((raw[:, :1], raw[:, 1 + modes:]), dim=1)
            pieces[-1].append(torch.einsum("bc,bcnd->bnd", error[start:end], carrier))
            if depth > 1:
                del raw
            del carrier
        for boundary in range(depth - 1):
            diagnostics[f"boundary{boundary}_coefficient_rms"] = math.sqrt(
                diagnostics.pop(f"boundary{boundary}_coefficient_mean_square"))
        return [torch.cat(part, dim=0) for part in pieces], diagnostics
