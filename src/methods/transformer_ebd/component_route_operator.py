"""Carrier, trace fitting and independent acceptance probes."""


from __future__ import annotations


import math


import torch


from .forward_response_credit import forward_vector_responses


from .token_complete_dfa import layer_norm_input_vjp


@torch.no_grad()
def identity_class_operator(model, caches):
    """Return M [B,C,N,D]; contraction with e equals identity-path credit."""
    final = caches[-1].block_output
    batch, tokens, width = final.shape
    classes = model.config.num_classes
    source = model.head.weight[None].expand(batch, classes, width)
    values = final[:, :1].expand_as(source)
    cls = layer_norm_input_vjp(
        values, source, model.final_norm.weight, model.final_norm.eps
    )
    operator = final.new_zeros(batch, classes, tokens, width)
    operator[:, :, 0] = cls
    return operator


@torch.no_grad()
def operator_response(operator, probes):
    """B^T p in row storage: [B,C,N,D], [B,P,N,D] -> [B,P,C]."""
    return torch.einsum("bcnd,bpnd->bpc", operator, probes)


@torch.no_grad()
def select_operator(carrier, candidate, probes, responses):
    """Forward validation decision independent of e; not a descent certificate."""
    baseline_error = (responses - operator_response(carrier, probes)).square().mean((1, 2))
    candidate_error = (responses - operator_response(candidate, probes)).square().mean((1, 2))
    accepted = candidate_error < baseline_error
    selected = torch.where(accepted[:, None, None, None], candidate, carrier)
    return selected, accepted


@torch.no_grad()
def sample_output_codes(reference, count, generator):
    """[B,K,C] codes with E[sum_k v_k v_k^T/K]=I, independent of error."""
    batch, classes = reference.shape[:2]
    if not 1 <= count <= classes:
        raise ValueError("Output probe count must be between 1 and class count")
    if count == classes:
        return (torch.eye(classes, device=reference.device, dtype=reference.dtype)
                * math.sqrt(classes))[None].expand(batch, -1, -1)
    raw = torch.randn(batch, classes, count, device=reference.device,
                      dtype=reference.dtype, generator=generator)
    return torch.linalg.qr(raw, mode="reduced").Q.transpose(1, 2) * math.sqrt(classes)


@torch.no_grad()
def batched_vector_responses(suffix, tokens, probes, epsilon):
    """Batch positive/negative class probes for a batch-separable suffix.

    Applicable to this ViT (LayerNorm, no dropout/BatchNorm). Each logical
    directional query is unchanged; this function trades activation memory
    for throughput. Do not use for sample-coupled/stochastic suffixes.
    """
    if epsilon <= 0 or probes.ndim != 4 or probes.shape[0] != tokens.shape[0] or probes.shape[2:] != tokens.shape[1:]:
        raise ValueError("Invalid batched forward response inputs")
    perturbed = torch.cat((tokens[:, None] + epsilon * probes,
                           tokens[:, None] - epsilon * probes), dim=1)
    batch, queries, count, width = perturbed.shape
    output = suffix(perturbed.reshape(batch * queries, count, width))
    output = output.reshape(batch, queries, -1)
    positive, negative = output.chunk(2, dim=1)
    return (positive - negative) / (2 * epsilon)


@torch.no_grad()
def trace_component_operator(suffix, tokens, carrier, dictionary, codes, *, epsilon=1e-3, ridge=0.1, trust_ratio=0.5, batch_output_queries=False):
    """Identify component coefficients using their EXACT known operator Gram.

    G_rs=<T_r,T_s>_F is available without downstream queries. The unknown
    b_r=<T_r,J^T-M>_F is estimated by averaging
        v^T [J (T_r v) - M^T (T_r v)].
    Queries therefore follow component paths and randomize in C-dimensional
    output space, rather than randomizing in ND-dimensional token space.
    The same v is used for all r. C orthogonal codes give the exact b in the
    infinitesimal-response limit. This still has 2*R*K suffix-forward cost.

    Exact b yields a ridge-shrunken orthogonal projection in OPERATOR space;
    Frobenius error cannot increase, including the scalar trust shrink below.
    Sampled codes/finite differences have noise; no such pointwise guarantee
    is asserted for them or for parameter updates after error contraction.
    """
    if epsilon <= 0 or ridge <= 0 or trust_ratio < 0:
        raise ValueError("epsilon/ridge must be positive; trust ratio nonnegative")
    if codes.ndim != 3 or codes.shape[0] != carrier.shape[0] or codes.shape[2] != carrier.shape[1] or codes.shape[1] < 1:
        raise ValueError("Codes must have shape [batch, output probes, classes]")
    flat = dictionary.flatten(2)
    gram = flat @ flat.transpose(1, 2)
    rhs = []
    for component in dictionary.unbind(1):
        raw = torch.einsum("bkc,bcnd->bknd", codes, component)
        # Unit RMS perturbations keep epsilon comparable across components.
        scale = raw.flatten(2).square().mean(-1).sqrt().clamp_min(1e-20)
        probes = raw / scale[:, :, None, None]
        measure = batched_vector_responses if batch_output_queries else forward_vector_responses
        response = measure(suffix, tokens, probes, epsilon)
        residual = (response - operator_response(carrier, probes)) * scale[:, :, None]
        rhs.append((residual * codes).sum(-1).mean(-1))
    rhs = torch.stack(rhs, dim=1)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram + ridge * scale[:, None, None] * identity, rhs.unsqueeze(-1)).squeeze(-1)
    correction = torch.einsum("br,brcnd->bcnd", coefficients, dictionary)
    limit = trust_ratio * carrier.flatten(1).norm(dim=-1)
    shrink = (limit / correction.flatten(1).norm(dim=-1).clamp_min(1e-20)).clamp(max=1)
    coefficients = coefficients * shrink[:, None]
    selected = carrier + correction * shrink[:, None, None, None]
    return selected, coefficients, rhs
