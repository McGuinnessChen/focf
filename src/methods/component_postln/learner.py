"""Common model, loss, Adam and trainable scope; change only feedback rules."""
from dataclasses import dataclass, fields
import math

import torch

from methods.grape.learner import GrAPEConfig, objective, output_derivative
from methods.grape_vit_stem_audit.learner import GroupedStemGrAPELearner
from methods.transformer_ebd.component_route_operator import identity_class_operator
from .feedback import ComponentFeedback
from .local_gradients import populate_local_gradients


@dataclass(frozen=True)
class CommonConfig(GrAPEConfig):
    parameter_scope: str = "all"
    feedback_column_lr: float = 0.1
    output_probes: int = 0
    operator_ridge: float = 0.1
    mirror_warmup: int = 16
    mirror_beta: float = 0.9
    mirror_probe_count: int = 16
    probe_epsilon: float = 1e-3
    trust_ratio: float = 0.5
    validation_probes: int = 8

    def __post_init__(self):
        base = {field.name: getattr(self, field.name) for field in fields(GrAPEConfig)}
        if self.method in ("component", "identity"):
            base["method"] = "dfa"
        GrAPEConfig(**base)
        if self.parameter_scope != "all":
            raise ValueError("Matched comparison requires all parameters trainable")
        if self.output_probes < 0 or self.mirror_warmup < 1 or self.mirror_probe_count < 2 or self.validation_probes < 1:
            raise ValueError("Invalid probe count")
        if not all(math.isfinite(v) for v in (self.mirror_beta, self.probe_epsilon, self.trust_ratio, self.operator_ridge)):
            raise ValueError("Nonfinite feedback configuration")
        if not 0 <= self.mirror_beta < 1 or self.probe_epsilon <= 0 or self.operator_ridge <= 0 or self.trust_ratio < 0:
            raise ValueError("Invalid feedback configuration")


class CommonLearner(GroupedStemGrAPELearner):
    def __init__(self, model, config):
        if model.norm_layout != "post":
            raise ValueError("This migration requires post-LN")
        super().__init__(model, config)
        self.component = ComponentFeedback(model, config) if config.method == "component" else None

    def step(self, images, labels):
        if self.config.method not in ("component", "identity"):
            return super().step(images, labels)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            cache = self.model.forward_with_cache(images)
            logits = cache.logits
            loss = objective(logits, labels, self.config.loss)
            error = output_derivative(logits, labels, self.config.loss)
            if self.component is not None:
                credits, diagnostics = self.component.prepare(cache.block_caches, error)
            else:
                carrier = identity_class_operator(self.model, cache.block_caches)
                credit = torch.einsum("bc,bcnd->bnd", error, carrier)
                credits = [credit for _ in self.model.blocks]
                diagnostics = {"local_mirror_forward_calls": 0, "suffix_forward_calls": 0,
                               "suffix_directional_evaluations": 0}
            populate_local_gradients(self.model, images, cache, error, credits)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            total = loss.new_zeros(())
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is None:
                    raise RuntimeError(f"Missing gradient for {name}")
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f"Nonfinite gradient for {name}")
                    total += p.grad.square().sum()
            diagnostics["parameter_gradient_norm"] = float(total.sqrt())
            diagnostics.update(self.model.activation_diagnostics)
            self.optimizer.step()
        self.steps += 1
        return logits.detach(), float(loss), diagnostics

    def calibrate(self, images, labels):
        if self.config.method in ("component", "identity"):
            raise ValueError("Component and identity methods prohibit BP calibration")
        return super().calibrate(images, labels)

    def state_dict(self):
        state = super().state_dict()
        state["component"] = self.component.state_dict() if self.component is not None else None
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self.component is not None:
            self.component.load_state_dict(state["component"])
