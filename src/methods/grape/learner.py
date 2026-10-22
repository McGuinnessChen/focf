"""Macro GrAPE adapted to ObservableViT, with isolated local reverse graphs.

All suffix JVPs and all local gradients precede a single model optimizer step.
Compared with EBD, feedback is learned from a Jacobian alignment objective,
not an error/activity EMA. Model updates use an ordinary PyTorch optimizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from .core import batch_rank_one_estimate, column_cosine_refinement, forward_jvp, project_error


@dataclass(frozen=True)
class GrAPEConfig:
    method: str = "grape"
    loss: str = "cross_entropy"
    parameter_scope: str = "blocks_head"
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # eta_B = column_learning_rate * hidden_dimension. Eq.17 retains 1/n.
    feedback_column_lr: float = 0.01
    aggregation: str = "mean_estimate"
    feedback_seed: int = 100
    calibration_lr: float = 0.01
    calibration_momentum: float = 0.9

    def __post_init__(self):
        if self.method not in ("grape", "dfa", "bp"):
            raise ValueError("Unknown training method")
        if self.loss not in ("cross_entropy", "mse"):
            raise ValueError("Unknown loss")
        if self.parameter_scope not in ("blocks_head", "all"):
            raise ValueError("Unknown parameter scope")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("Unknown optimizer")
        if self.aggregation not in ("mean_estimate", "mean_sample_cosine"):
            raise ValueError("Unknown batch alignment convention")
        for value in (self.learning_rate, self.weight_decay, self.feedback_column_lr, self.calibration_lr):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Learning rates and decay must be finite and nonnegative")


def objective(logits, labels, name):
    if name == "cross_entropy":
        return F.cross_entropy(logits, labels)
    target = F.one_hot(labels, logits.shape[1]).to(logits)
    return F.mse_loss(logits, target)


@torch.no_grad()
def output_derivative(logits, labels, name):
    target = F.one_hot(labels, logits.shape[1]).to(logits)
    if name == "cross_entropy":
        return (logits.softmax(-1) - target) / logits.shape[0]
    return 2 * (logits - target) / logits.numel()


@torch.no_grad()
def sample_cosine_refinement(feedback, probe, response, column_lr, epsilon=1e-12):
    """Mean of sample cosine gradients, without materializing B*C*N*D.

    A column of each sample's rank-one estimator is p_k*g. Its normalized
    direction is sign(p_k)*g/||g||. Preserve the mean's magnitude: normalizing
    it again would silently replace this objective with a different one.
    """
    p = probe.flatten(1)
    g = response / response.norm(dim=1, keepdim=True).clamp_min(epsilon)
    target_mean = g.T @ p.sign() / p.shape[0]
    norm = feedback.norm(dim=0, keepdim=True).clamp_min(epsilon)
    cosine = (feedback / norm * target_mean).sum(0, keepdim=True)
    updated = feedback + column_lr * (target_mean / norm - cosine * feedback / norm.square())
    return updated / (updated.norm(dim=0, keepdim=True) + epsilon)


class MacroGrAPELearner:
    def __init__(self, model, config: GrAPEConfig):
        self.model, self.config = model, config
        if model.config.readout_mode != "standard":
            raise ValueError("GrAPE adapter requires the standard final readout")
        if config.parameter_scope == "blocks_head":
            model.enable_protocol_matched_bp_learning()
        else:
            model.enable_full_bp_learning()
        params = [p for p in model.parameters() if p.requires_grad]
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum,
                                             weight_decay=config.weight_decay)
        # Paper's calibration changes model parameters, not feedback matrices.
        # Keep its SGD momentum distinct; record this adaptation explicitly.
        self.calibration_optimizer = torch.optim.SGD(params, lr=config.calibration_lr,
                                                     momentum=config.calibration_momentum,
                                                     weight_decay=config.weight_decay)
        reference = model.head.weight
        self.generator = torch.Generator(device=reference.device).manual_seed(config.feedback_seed)
        self.boundary_indices = ([-1] if config.parameter_scope == "all" else []) + list(range(model.config.depth))
        self.feedback = []
        for _ in self.boundary_indices:
            a = torch.randn(model.config.num_classes, model.config.token_count * model.config.embed_dim,
                            generator=self.generator, device=reference.device, dtype=reference.dtype)
            a /= a.norm(dim=0, keepdim=True) + 1e-12
            self.feedback.append(a)
        self.initial_feedback = [a.clone() for a in self.feedback]
        self.steps = 0
        self.calibration_batches = 0

    def suffix(self, boundary, hidden):
        for block in self.model.blocks[boundary + 1:]:
            hidden, _ = block(hidden)
        return self.model.head(self.model.final_norm(hidden)[:, 0])

    @torch.no_grad()
    def refine_feedback(self, cache):
        diagnostics = {}
        for index, boundary in enumerate(self.boundary_indices):
            h = cache.patch_tokens if boundary == -1 else cache.block_caches[boundary].block_output
            a = self.feedback[index]
            if self.config.method == "grape":
                p = torch.randn(h.shape, generator=self.generator, device=h.device, dtype=h.dtype)
                _, g = forward_jvp(lambda z: self.suffix(boundary, z), h, p)
                estimate = batch_rank_one_estimate(p, g)
                if self.config.aggregation == "mean_estimate":
                    updated = column_cosine_refinement(
                        a, estimate, learning_rate=self.config.feedback_column_lr * a.shape[1],
                    )
                else:
                    updated = sample_cosine_refinement(a, p, g, self.config.feedback_column_lr)
                direction = estimate / estimate.norm(dim=0, keepdim=True).clamp_min(1e-12)
                diagnostics[f"boundary{boundary}_estimate_cosine_before"] = float((a * direction).sum(0).mean())
                diagnostics[f"boundary{boundary}_estimate_cosine_after"] = float((updated * direction).sum(0).mean())
                diagnostics[f"boundary{boundary}_relative_feedback_change"] = float((updated - a).norm() / a.norm())
                self.feedback[index] = updated
            diagnostics[f"boundary{boundary}_initial_feedback_cosine"] = float(
                (self.feedback[index] * self.initial_feedback[index]).sum(0).mean())
        diagnostics["suffix_jvp_calls"] = len(self.boundary_indices) if self.config.method == "grape" else 0
        diagnostics["suffix_block_evaluations"] = sum(self.model.config.depth - b - 1 for b in self.boundary_indices) if self.config.method == "grape" else 0
        return diagnostics

    @staticmethod
    def local_parameter_gradients(output, parameters, credit):
        parameters = [p for p in parameters if p.requires_grad]
        if not parameters:
            return
        gradients = torch.autograd.grad(output, parameters, grad_outputs=credit.detach())
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.detach()

    def populate_local_gradients(self, images, cache, error, credits):
        """One graph per macro module; detached inputs prohibit cross-block BP."""
        credit_offset = 0
        if self.config.parameter_scope == "all":
            tokens = self.model.tokenize(images.detach())
            parameters = list(self.model.patch_embed.parameters()) + [self.model.cls_token, self.model.position]
            self.local_parameter_gradients(tokens, parameters, credits[0])
            credit_offset = 1
        for index, block in enumerate(self.model.blocks):
            # Recompute only this block; later blocks cannot enter this graph.
            output, _ = block(cache.block_caches[index].block_input.detach())
            self.local_parameter_gradients(output, block.parameters(), credits[index + credit_offset])
        last = cache.block_caches[-1].block_output.detach()
        logits = self.model.head(self.model.final_norm(last)[:, 0])
        self.local_parameter_gradients(
            logits, list(self.model.final_norm.parameters()) + list(self.model.head.parameters()), error,
        )

    def step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.config.method == "bp":
            logits = self.model(images)
            loss = objective(logits, labels, self.config.loss)
            loss.backward()
            diagnostics = {"suffix_jvp_calls": 0, "suffix_block_evaluations": 0}
        else:
            with torch.no_grad():
                cache = self.model.forward_with_cache(images)
                logits = cache.logits
                loss = objective(logits, labels, self.config.loss)
                error = output_derivative(logits, labels, self.config.loss)
                diagnostics = self.refine_feedback(cache)
                credits = [project_error(a, error).reshape(images.shape[0], self.model.config.token_count,
                                                           self.model.config.embed_dim) for a in self.feedback]
            self.populate_local_gradients(images, cache, error, credits)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        total_norm_sq = loss.new_zeros(())
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise RuntimeError(f"Missing gradient for {name}")
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient for {name}")
                total_norm_sq += parameter.grad.square().sum()
        diagnostics["parameter_gradient_norm"] = float(total_norm_sq.sqrt())
        self.optimizer.step()
        self.steps += 1
        return logits.detach(), float(loss.detach()), diagnostics

    def calibrate(self, images, labels):
        if self.config.method == "bp":
            raise ValueError("BP reference does not need extra calibration")
        self.model.train()
        self.calibration_optimizer.zero_grad(set_to_none=True)
        loss = objective(self.model(images), labels, self.config.loss)
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError("Nonfinite calibration gradient")
        self.calibration_optimizer.step()
        self.calibration_batches += 1
        return float(loss.detach())

    def state_dict(self):
        return {"config": asdict(self.config), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "calibration_optimizer": self.calibration_optimizer.state_dict(),
                "feedback": self.feedback, "initial_feedback": self.initial_feedback,
                "probe_rng": self.generator.get_state(), "steps": self.steps,
                "calibration_batches": self.calibration_batches}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("Checkpoint learning configuration differs")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.calibration_optimizer.load_state_dict(state["calibration_optimizer"])
        reference = self.model.head.weight
        self.feedback = [a.to(reference) for a in state["feedback"]]
        self.initial_feedback = [a.to(reference) for a in state["initial_feedback"]]
        self.generator.set_state(state["probe_rng"].cpu())
        self.steps, self.calibration_batches = state["steps"], state["calibration_batches"]
