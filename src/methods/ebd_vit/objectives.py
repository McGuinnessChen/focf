"""EBD local activity objectives, with explicit batch and token reductions.

E is the per-example output error, H[b,n,d] the block output. Correlation
uses every token, exactly as a flattened hidden layer. For tractable entropy,
R[n,:,:] is a per-token channel second moment (a block-diagonal covariance
approximation over the flattened activity). This is an explicit ViT adaptation.

No normalization of correlation matrices or gradients is performed here.
The EMA derivative factors (1-beta) are absorbed in the objective coefficients,
as in the author's implementation. All regularizer choices remain configurable.
"""
from dataclasses import dataclass, asdict
import math

import torch


@dataclass(frozen=True)
class EBDObjectiveConfig:
    correlation_beta: float = 0.999999
    covariance_beta: float = 0.99999
    correlation_gain: float = 0.01
    covariance_initial: float = 0.03
    entropy_epsilon: float = 1e-6
    error_scale: float = 1.0
    entropy_scale: float = 0.0015
    power_scale: float = 0.006
    l1_scale: float = 0.3
    power_target: float = 1.0
    power_mode: str = "author_sample"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "power_mode":
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.correlation_beta < 1 or not 0 <= self.covariance_beta < 1:
            raise ValueError("EMA coefficients must be in [0,1)")
        if self.covariance_initial <= 0 or self.entropy_epsilon <= 0 or self.power_target <= 0:
            raise ValueError("Covariance initialization, entropy ridge and power target must be positive")
        if self.power_mode not in ("author_sample", "paper_batch"):
            raise ValueError("Unknown power objective")


class EBDActivityObjective:
    def __init__(self, classes, tokens, width, config, *, generator, reference):
        self.config = config
        self.correlation = torch.randn(classes, tokens, width, generator=generator,
            device=reference.device, dtype=reference.dtype) * config.correlation_gain
        self.covariance = torch.eye(width, device=reference.device, dtype=reference.dtype)[None].repeat(tokens,1,1)
        self.covariance.mul_(config.covariance_initial)
        self.updates = 0

    @torch.no_grad()
    def update(self, error, activity):
        if activity.ndim != 3 or error.ndim != 2 or len(activity) != len(error):
            raise ValueError("Expected paired [B,C] errors and [B,N,D] activities")
        expected = (error.shape[1], activity.shape[1], activity.shape[2])
        if self.correlation.shape != expected or len(error) == 0:
            raise ValueError("Activity / output dimensions differ from EBD state")
        b = len(error)
        # Author convention R_eh = EMA(E^T H / B), with token identity retained.
        c = torch.einsum("bc,bnd->cnd", error, activity) / b
        r = torch.einsum("bni,bnj->nij", activity, activity) / b
        cfg = self.config
        self.correlation.mul_(cfg.correlation_beta).add_(c, alpha=1-cfg.correlation_beta)
        self.covariance.mul_(cfg.covariance_beta).add_(r, alpha=1-cfg.covariance_beta)
        self.updates += 1

    @torch.no_grad()
    def credit(self, error, activity):
        """Return dJ/dH after updating statistics; includes exactly one /B."""
        b, _, d = activity.shape
        cfg = self.config
        corr = torch.einsum("bc,cnd->bnd", error, self.correlation) / b
        entropy = torch.zeros_like(activity)
        if cfg.entropy_scale:
            # Solve R_n X_n = H_n^T; do not invert a flattened (N*D)^2 matrix.
            regularized = self.covariance + cfg.entropy_epsilon * torch.eye(
                d, device=activity.device, dtype=activity.dtype)[None]
            whitened = torch.linalg.solve(regularized, activity.permute(1,2,0)).permute(2,0,1)
            entropy = -2 * whitened / (b*d)
        if cfg.power_mode == "author_sample":
            power = (activity.square() - cfg.power_target) * activity / b
        else:
            power = (activity.square().mean(0,keepdim=True) - cfg.power_target) * activity / b
        sparse = activity.sign() / (b*d)
        credit = (cfg.error_scale*corr + cfg.entropy_scale*entropy
                  + cfg.power_scale*power + cfg.l1_scale*sparse)
        rms = lambda x: float(x.square().mean().sqrt())
        diagnostics = {"correlation_norm":float(self.correlation.norm()),
            "activity_rms":rms(activity), "activity_batch_variance":float(activity.var(0,unbiased=False).mean()),
            "correlation_credit_rms":rms(cfg.error_scale*corr),
            "entropy_credit_rms":rms(cfg.entropy_scale*entropy),
            "power_credit_rms":rms(cfg.power_scale*power),
            "l1_credit_rms":rms(cfg.l1_scale*sparse), "total_credit_rms":rms(credit)}
        return credit, diagnostics

    def state_dict(self):
        return {"correlation":self.correlation, "covariance":self.covariance,
                "updates":self.updates, "config":asdict(self.config)}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("EBD objective configuration changed on resume")
        for name in ("correlation", "covariance"):
            old = getattr(self,name)
            new = state[name]
            if new.shape != old.shape:
                raise ValueError(f"Invalid {name} shape on resume")
            old.copy_(new.to(old))
        self.updates = int(state["updates"])
