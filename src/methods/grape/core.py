"""GrAPE's activation-space estimator and Appendix C alignment update.

Reference: Caillon et al., ICLR 2026, Algorithm 1 and Appendix C, Eq. (17).
https://proceedings.iclr.cc/paper_files/paper/2026/file/2b1c608556ce011d3a3381fe0c527136-Paper-Conference.pdf

We store A with the Jacobian's shape [classes, hidden coordinates]. Thus row
errors project as e @ A, or column errors as A.T @ e. This resolves the paper's
transposed B notation between the alignment and feedback equations. Columns
are hidden coordinates, NOT classes. No EBD correlation EMA, RMS-normalized
parameter update, or component correction is introduced here.

These are independently checked primitives, not a reproduction of the paper's
reported benchmarks. Batch aggregation and the ViT partition remain explicit
adaptation choices to be recorded by a future training runner.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import torch


@torch.no_grad()
def forward_jvp(
    suffix: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    probe: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One isolated activation perturbation using genuine forward-mode AD.

    The callable must be deterministic for this evaluation. For per-example
    Jacobians it must not couple samples (our ViT uses LayerNorm, not BatchNorm).
    Parameters are held fixed; no reverse-mode fallback is allowed.
    """
    if hidden.shape != probe.shape:
        raise ValueError("Hidden state and probe must have identical shapes")
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(hidden.detach(), probe.detach())
        primal, tangent = torch.autograd.forward_ad.unpack_dual(suffix(dual))
        if tangent is None:
            tangent = torch.zeros_like(primal)
        return primal.detach(), tangent.detach()


@torch.no_grad()
def batch_rank_one_estimate(probe: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Return mean_b (J_b p_b) p_b.T, shape [classes, hidden coordinates].

    One independent isotropic Gaussian probe per example yields a conditional
    unbiased estimate of the mean batch Jacobian. This function does not
    normalize probes: doing so would change the unbiased estimator's scale.
    Averaging estimators BEFORE cosine is an explicit batch convention; it is
    not equal to averaging individual cosine gradients.
    """
    if probe.ndim < 2 or response.ndim != 2 or probe.shape[0] != response.shape[0]:
        raise ValueError("Expected probes [batch,...] and responses [batch,classes]")
    if probe.shape[0] == 0:
        raise ValueError("Empty batches are not supported")
    return response.T @ probe.flatten(1) / probe.shape[0]


@torch.no_grad()
def column_cosine_refinement(
    feedback: torch.Tensor,
    estimate: torch.Tensor,
    *,
    learning_rate: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """One analytical descent step on 1-mean_k cos(A[:,k], Jhat[:,k]).

    Implements Eq. (17), including its 1/n_hidden factor, then column
    normalization. Norms in the cosine use clamp_min(epsilon); their derivative
    is zero below the clamp. Final normalization uses norm + epsilon as in
    Section 3.3. Zero-response columns contribute zero alignment gradient.
    Returns new storage, leaving the caller's feedback untouched.
    """
    if feedback.ndim != 2 or feedback.shape != estimate.shape or feedback.numel() == 0:
        raise ValueError("Feedback and estimate must be matching nonempty matrices")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    norm = feedback.norm(dim=0, keepdim=True)
    safe_norm = norm.clamp_min(epsilon)
    target = estimate / estimate.norm(dim=0, keepdim=True).clamp_min(epsilon)
    cosine = ((feedback / safe_norm) * target).sum(dim=0, keepdim=True)
    # This gradient improves an estimate-alignment objective. It is not EBD's
    # error/activity correlation update, and does not use labels or output error.
    ascent = target / safe_norm - (
        cosine * feedback / safe_norm.square() * (norm >= epsilon)
    )
    updated = feedback + (learning_rate / feedback.shape[1]) * ascent
    return updated / (updated.norm(dim=0, keepdim=True) + epsilon)


@torch.no_grad()
def project_error(feedback: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """Project row output derivatives [batch,classes] to [batch,hidden].

    Loss reduction belongs to the caller. Do not divide by batch size here if
    the supplied derivative already corresponds to a mean-reduced loss.
    """
    if feedback.ndim != 2 or error.ndim != 2 or feedback.shape[0] != error.shape[1]:
        raise ValueError("Expected feedback [classes,hidden] and error [batch,classes]")
    return error @ feedback


def calibration_due(completed_epoch: int, interval: int) -> bool:
    """End-of-epoch schedule: interval=0 disables BP; T>0 adds one batch."""
    if not isinstance(completed_epoch, int) or completed_epoch < 1:
        raise ValueError("completed_epoch must be a positive integer")
    if not isinstance(interval, int) or interval < 0:
        raise ValueError("interval must be a nonnegative integer")
    return interval > 0 and completed_epoch % interval == 0
