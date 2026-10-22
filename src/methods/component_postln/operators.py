"""Forward-observable component dictionary for post-LN encoder blocks.

EBD notation: q_l = B_l(s) e, B_l(s) = M(s) + sum_r alpha_lr T_lr(s).
Only forward responses identify alpha; e does not enter that identification.
Unlike pre-LN, the residual path itself contains two normalization Jacobians.
"""
import math

import torch

from methods.transformer_ebd.component_route_operator import identity_class_operator
from methods.transformer_ebd.ffn_ebd import gelu_derivative
from methods.transformer_ebd.token_complete_dfa import layer_norm_input_vjp


def ln_adjoint(values, modulation, layer):
    return layer_norm_input_vjp(values[:, None].expand_as(modulation), modulation,
                                layer.weight, layer.eps)


@torch.no_grad()
def block_component_operators(carrier, cache, block, maps):
    """M + sum(parts) is the exact block adjoint with oracle channel maps.

    For u = N1(x + A(x)), y = N2(u + F(u)):
      J^T M = (I + A^T) N1^T (I + F^T) N2^T M.
    Training supplies forward-mirrored maps, never the oracle channel weights.
    The normalization increment is essential: post-LN has no pure identity
    residual path. LayerNorm scales and cached forward states are observable.
    """
    r1 = cache.block_input + cache.attention_branch
    r2 = cache.after_attention + cache.ffn_branch
    v = ln_adjoint(r2, carrier, block.ln2)
    norm_path = ln_adjoint(r1, v, block.ln1)
    ffn_hidden = (v @ maps.ffn_output) * gelu_derivative(cache.ffn_preactivation)[:, None]
    ffn = ln_adjoint(r1, ffn_hidden @ maps.ffn_input, block.ln1)
    attention_source = norm_path + ffn
    batch, classes, tokens, _ = carrier.shape
    heads, head_dim = block.attention.num_heads, block.attention.head_dim
    context = (attention_source @ maps.output).reshape(batch, classes, tokens, heads, head_dim)
    context = context.permute(0, 1, 3, 2, 4)
    raw_score = torch.einsum("bchqd,bhkd->bchqk", context, cache.v)
    attention = cache.attention[:, None]
    score = attention * (raw_score - (attention * raw_score).sum(-1, keepdim=True))
    query = (score @ cache.k[:, None]) / math.sqrt(head_dim)
    key = (score.transpose(-1, -2) @ cache.q[:, None]) / math.sqrt(head_dim)
    value = attention.transpose(-1, -2) @ context
    parts, names = [norm_path - carrier, ffn], ["normalization", "ffn"]
    for head in range(heads):
        for role, signal, channel_map in (("query", query, maps.query),
                                          ("key", key, maps.key),
                                          ("value", value, maps.value)):
            parts.append(signal[:, :, head] @ channel_map[head])
            names.append(f"head{head}_{role}")
    return torch.stack(parts, dim=1), names


@torch.no_grad()
def downstream_component_dictionary(model, caches, maps, boundary):
    if not 0 <= boundary < len(caches):
        raise ValueError("Invalid block boundary")
    carrier = identity_class_operator(model, caches)
    families, names = [carrier[:, None]], ["carrier_scale"]
    for index in range(boundary + 1, len(caches)):
        parts, part_names = block_component_operators(carrier, caches[index], model.blocks[index], maps[index])
        families.append(parts)
        names.extend(f"block{index}_{name}" for name in part_names)
    raw = torch.cat(families, dim=1)
    scale = carrier.flatten(1).norm(dim=-1)[:, None] / raw.flatten(2).norm(dim=-1).clamp_min(1e-20)
    return carrier, raw * scale[:, :, None, None, None], names
