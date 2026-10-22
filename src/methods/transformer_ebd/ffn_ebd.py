"""Analytic GELU derivative."""


from __future__ import annotations


import math


import torch


def gelu_derivative(preactivation: torch.Tensor) -> torch.Tensor:
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    normal_pdf = torch.exp(-0.5 * preactivation.square()) / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(preactivation * inv_sqrt_two)) + (
        preactivation * normal_pdf
    )
