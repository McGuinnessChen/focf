"""Block EBD: local activity objectives and exact within-block derivatives.

The feedback boundary is identical to block DFA / the matched GrAPE baseline.
This is NOT the old component EBD path, and NOT author-supplied ViT code.
No forward mirror, downstream routing estimator or periodic BP calibration is
used. The tokenizer belongs to block 0. Every model parameter is trainable.
"""
from dataclasses import dataclass, asdict, field
import math

import torch
from torch.nn import functional as F

from methods.component_postln.local_gradients import populate_local_gradients
from .objectives import EBDActivityObjective, EBDObjectiveConfig


def supervised_objective(logits, labels, loss):
    if loss == "cross_entropy":
        return F.cross_entropy(logits, labels)
    if loss == "mse":
        target = F.one_hot(labels, logits.shape[-1]).to(logits)
        # The original EBD residual is d(0.5 * sum_c (z-y)^2)/dz.
        return .5*(logits-target).square().sum(-1).mean()
    raise ValueError("Unknown supervised objective")


@torch.no_grad()
def example_error(logits, labels, loss):
    target = F.one_hot(labels, logits.shape[-1]).to(logits)
    if loss == "cross_entropy":
        return logits.softmax(-1)-target
    if loss == "mse":
        return logits-target
    raise ValueError("Unknown supervised objective")


@dataclass(frozen=True)
class EBDConfig:
    loss: str = "cross_entropy"
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    momentum: float = .9
    weight_decay: float = 1e-4
    feedback_seed: int = 100
    objective: EBDObjectiveConfig = field(default_factory=EBDObjectiveConfig)

    def __post_init__(self):
        if self.loss not in ("cross_entropy", "mse") or self.optimizer not in ("adam","sgd"):
            raise ValueError("Unsupported loss/optimizer")
        for value in (self.learning_rate,self.momentum,self.weight_decay):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Nonfinite/negative optimizer configuration")
        if self.momentum >= 1:
            raise ValueError("Momentum must be below one")


class EBDLearner:
    def __init__(self, model, config):
        if model.norm_layout != "post":
            raise ValueError("Block EBD currently supports the matched post-LN architecture")
        self.model, self.config = model, config
        model.enable_full_bp_learning()  # Sets requires_grad; training below uses analytic local VJPs.
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Matched EBD requires all parameters trainable")
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                               weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                momentum=config.momentum, weight_decay=config.weight_decay)
        reference = model.head.weight
        generator = torch.Generator(device=reference.device).manual_seed(config.feedback_seed)
        self.objectives = [EBDActivityObjective(model.config.num_classes, model.config.token_count,
            model.config.embed_dim, config.objective, generator=generator, reference=reference)
            for _ in model.blocks]
        self.steps = 0
        self.calibration_batches = 0

    @torch.no_grad()
    def step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        cache = self.model.forward_with_cache(images)
        logits = cache.logits
        loss = supervised_objective(logits,labels,self.config.loss)
        error = example_error(logits,labels,self.config.loss)
        credits, diagnostics = [], {}
        for index,(state,block) in enumerate(zip(self.objectives,cache.block_caches)):
            state.update(error,block.block_output)
            credit, info = state.credit(error,block.block_output)
            credits.append(credit)
            diagnostics.update({f"block{index}_{name}":value for name,value in info.items()})
        # Only local VJPs are shared with the common model infrastructure; no
        # component feedback mechanism or downstream Jacobian enters this step.
        populate_local_gradients(self.model,images,cache,error/len(labels),credits)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite EBD supervised loss")
        norm = loss.new_zeros(())
        for name,p in self.model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Missing/nonfinite EBD local gradient: {name}")
            norm += p.grad.square().sum()
        diagnostics["parameter_gradient_norm"] = float(norm.sqrt())
        self.optimizer.step()
        self.steps += 1
        return logits.detach(),float(loss),diagnostics

    def state_dict(self):
        return {"model":self.model.state_dict(), "optimizer":self.optimizer.state_dict(),
            "objectives":[o.state_dict() for o in self.objectives], "steps":self.steps,
            "calibration_batches":0, "config":asdict(self.config)}

    def load_state_dict(self,state):
        if state["config"] != asdict(self.config) or len(state["objectives"]) != len(self.objectives):
            raise ValueError("EBD learning rule differs from checkpoint")
        if state["calibration_batches"] != 0:
            raise ValueError("EBD checkpoint cannot contain calibration")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for objective,saved in zip(self.objectives,state["objectives"]):
            objective.load_state_dict(saved)
        self.steps = int(state["steps"])
