"""Forward finite-difference responses and probe sampling."""


from __future__ import annotations


import math


from typing import Callable


import torch


@torch.no_grad()
def sample_probes(
    reference: torch.Tensor,
    count: int,
    generator: torch.Generator,
    *,
    orthogonal: bool,
) -> torch.Tensor:
    """Independent per-example full-token probes; E[p p^T] = I."""
    batch, tokens, width = reference.shape
    dimensions = tokens * width
    if not 0 < count <= dimensions:
        raise ValueError("Probe count must be between 1 and token dimension")
    raw = torch.randn(
        batch, dimensions, count, generator=generator,
        device=reference.device, dtype=reference.dtype,
    )
    if orthogonal:
        raw = torch.linalg.qr(raw, mode="reduced").Q * math.sqrt(dimensions)
    return raw.transpose(1, 2).reshape(batch, count, tokens, width)


@torch.no_grad()
def forward_vector_responses(
    suffix: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    probes: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return [batch, modes, classes] responses without contracting with e.

    Holding probes independent of the error lets each pair of forward calls
    constrain an entire broadcast operator, reusable for any output error at
    the same state. No extra forwards are required to retain all classes.
    Parameters and stochastic forward state must remain fixed across calls.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if probes.ndim != tokens.ndim + 1 or probes.shape[0] != tokens.shape[0] or probes.shape[2:] != tokens.shape[1:]:
        raise ValueError("Incompatible token and probe shapes")
    if probes.shape[1] < 1:
        raise ValueError("At least one probe is required")
    return torch.stack([
        (suffix(tokens + epsilon * probe) - suffix(tokens - epsilon * probe)) / (2 * epsilon)
        for probe in probes.unbind(dim=1)
    ], dim=1)
