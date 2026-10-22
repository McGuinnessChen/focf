"""Analytic layer-normalization input derivative."""


from __future__ import annotations


import torch


def layer_norm_input_vjp(
    input_values: torch.Tensor,
    output_modulation: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Explicit LayerNorm input VJP over the final feature dimension."""
    if input_values.shape != output_modulation.shape:
        raise ValueError("LayerNorm input and modulation shapes must match")
    if weight.shape != input_values.shape[-1:]:
        raise ValueError("LayerNorm affine weight has the wrong shape")
    centered = input_values - input_values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    inverse_std = torch.rsqrt(variance + float(eps))
    normalized = centered * inverse_std
    affine_modulation = output_modulation * weight
    mean_modulation = affine_modulation.mean(dim=-1, keepdim=True)
    mean_projection = (affine_modulation * normalized).mean(
        dim=-1, keepdim=True
    )
    return inverse_std * (
        affine_modulation
        - mean_modulation
        - normalized * mean_projection
    )
