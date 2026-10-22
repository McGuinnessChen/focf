"""Forward identification of local channel maps."""


from __future__ import annotations


import math


import torch


from torch import nn


def normalize_mirror_map(matrix: torch.Tensor) -> torch.Tensor:
    """Match the Frobenius budget of the corresponding orthogonal DFA map."""
    target_norm = math.sqrt(float(min(matrix.shape)))
    return matrix * (
        target_norm / (torch.linalg.vector_norm(matrix) + 1e-12)
    )


def estimate_linear_map_from_forward(
    module: nn.Module,
    probes: torch.Tensor,
    *,
    shuffle_responses: bool,
) -> torch.Tensor:
    """Estimate the row-layout linear map from local forward responses.

    For y=x W^T+b and white probes xi, E[(y(xi)-y(0))^T xi]=W.
    The returned [out,in] matrix therefore maps a row covector with q @ W,
    exactly the orientation needed by the local VJP. No parameter is read.
    """
    if probes.ndim != 2 or probes.shape[0] < 2:
        raise ValueError("Mirror probes must have shape [K,in] with K >= 2")
    baseline = module(probes.new_zeros(1, probes.shape[1]))
    responses = module(probes) - baseline
    if responses.ndim != 2 or responses.shape[0] != probes.shape[0]:
        raise ValueError("Mirrored module must return a [K,out] tensor")
    if shuffle_responses:
        responses = torch.roll(responses, shifts=1, dims=0)
    return responses.transpose(0, 1) @ probes / float(probes.shape[0])
