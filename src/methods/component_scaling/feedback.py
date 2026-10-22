"""Sample/query chunking preserves the exact class-space dictionary Gram.

EBD: q_l = [M + sum_r alpha_r T_r] e. Fitting never sees e. For K<C,
v=sqrt(C)*Q[:,k], E[sum_k v v^T/K]=I, so the finite-difference RHS targets
the same trace as the complete class basis in the infinitesimal limit.
This does not retain the full-basis pointwise improvement guarantee.
"""
from dataclasses import fields, replace
import math

import torch

from methods.component_postln.feedback import ComponentFeedback
from methods.component_postln.operators import downstream_component_dictionary
from methods.transformer_ebd.component_route_operator import (
    identity_class_operator, sample_output_codes, operator_response,
    select_operator, batched_vector_responses,
)
from methods.transformer_ebd.forward_response_credit import sample_probes


def slice_caches(caches, start, end):
    return [replace(cache, **{f.name: getattr(cache, f.name)[start:end]
                             for f in fields(cache) if isinstance(getattr(cache, f.name), torch.Tensor)})
            for cache in caches]


@torch.no_grad()
def chunked_vector_responses(suffix, tokens, probes, epsilon, query_chunk):
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    return torch.cat([batched_vector_responses(suffix, tokens, p, epsilon)
                      for p in probes.split(query_chunk, dim=1)], dim=1)


@torch.no_grad()
def trace_component_operator_chunked(suffix, tokens, carrier, dictionary, codes, *,
                                     epsilon, ridge, trust_ratio, query_chunk):
    """Exact Gram/normalization; stream only the RHS forward measurements.

    The dictionary remains independent of the output sketch. Using a sketch
    to construct BOTH the dictionary and its Gram would change this target.
    Positive and negative responses for each direction share one model state.
    """
    if epsilon <= 0 or ridge <= 0 or trust_ratio < 0 or query_chunk < 1:
        raise ValueError("Invalid trace configuration")
    if codes.ndim != 3 or codes.shape[0] != carrier.shape[0] or codes.shape[2] != carrier.shape[1] or codes.shape[1] < 1:
        raise ValueError("Codes must have shape [batch, queries, classes]")
    flat = dictionary.flatten(2)
    gram = flat @ flat.transpose(1, 2)
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
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram + ridge * scale[:, None, None] * identity,
                                      rhs.unsqueeze(-1)).squeeze(-1)
    correction = torch.einsum("br,brcnd->bcnd", coefficients, dictionary)
    limit = trust_ratio * carrier.flatten(1).norm(dim=-1)
    shrink = (limit / correction.flatten(1).norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
    return carrier + correction * shrink[:, None, None, None], coefficients * shrink[:, None], rhs


class ChunkedComponentFeedback(ComponentFeedback):
    @torch.no_grad()
    def prepare(self, caches, error):
        cfg = self.config
        batch = error.shape[0]
        count = cfg.output_probes or self.model.config.num_classes
        sample_chunk = min(cfg.feedback_sample_chunk, batch)
        # Exactly once per optimization minibatch, independent of chunk count.
        cycles = cfg.mirror_warmup if self.mirrors.observations == 0 else 1
        for _ in range(cycles):
            self.mirrors.update()
        diagnostics = {"local_mirror_forward_calls": cycles * 8 * (len(caches) - 1),
                       "suffix_forward_calls": 0, "suffix_directional_evaluations": 0,
                       "feedback_sample_chunks": math.ceil(batch / sample_chunk),
                       "feedback_output_probes": count}
        credits = []
        for boundary, cache in enumerate(caches):
            # Draw in the legacy boundary order for the WHOLE minibatch. This
            # isolates the statistical rule/RNG state from execution chunks.
            if boundary < len(caches) - 1:
                shape_reference = error.new_empty(batch, self.model.config.num_classes, 0)
                codes = sample_output_codes(shape_reference, count, self.generator)
                validation = sample_probes(cache.block_output, cfg.validation_probes,
                                            self.generator, orthogonal=True)
            pieces = []
            for start in range(0, batch, sample_chunk):
                end = min(start + sample_chunk, batch)
                subset = slice_caches(caches, start, end)
                if boundary == len(caches) - 1:
                    selected = identity_class_operator(self.model, subset)
                    pieces.append(torch.einsum("bc,bcnd->bnd", error[start:end], selected))
                    del selected
                    continue
                carrier, dictionary, _ = downstream_component_dictionary(
                    self.model, subset, self.mirrors.maps, boundary)

                def suffix(tokens):
                    diagnostics["suffix_forward_calls"] += 1
                    for block in self.model.blocks[boundary + 1:]:
                        tokens, _ = block(tokens)
                    return self.model.head(self.model.final_norm(tokens)[:, 0])

                candidate, coefficients, _ = trace_component_operator_chunked(
                    suffix, subset[boundary].block_output, carrier, dictionary, codes[start:end],
                    epsilon=cfg.probe_epsilon, ridge=cfg.operator_ridge,
                    trust_ratio=cfg.trust_ratio, query_chunk=cfg.output_query_chunk)
                response = chunked_vector_responses(
                    suffix, subset[boundary].block_output, validation[start:end],
                    cfg.probe_epsilon, cfg.output_query_chunk)
                selected, accepted = select_operator(carrier, candidate, validation[start:end], response)
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
                # Per-example logical evaluations, comparable across chunking.
                diagnostics["suffix_directional_evaluations"] += weight * 2 * (dictionary.shape[1] * count + cfg.validation_probes)
                pieces.append(torch.einsum("bc,bcnd->bnd", error[start:end], selected))
                del carrier, dictionary, candidate, coefficients, selected, response
            if boundary < len(caches) - 1:
                diagnostics[f"boundary{boundary}_coefficient_rms"] = math.sqrt(
                    diagnostics.pop(f"boundary{boundary}_coefficient_mean_square"))
            credits.append(torch.cat(pieces, dim=0))
        return credits, diagnostics
