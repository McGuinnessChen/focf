"""Post-LN FOCF: forward channel mirrors and error-independent trace fitting."""
from dataclasses import fields

import torch

from methods.transformer_ebd.component_route_operator import (
    identity_class_operator, sample_output_codes, trace_component_operator, select_operator,
)
from methods.transformer_ebd.forward_mirrored_feedback import (
    normalize_mirror_map, estimate_linear_map_from_forward,
)
from methods.transformer_ebd.forward_response_credit import sample_probes, forward_vector_responses
from methods.transformer_ebd.residual_path_feedback import make_residual_path_maps
from .operators import downstream_component_dictionary


class ForwardMirrorBank:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.maps = make_residual_path_maps(model, seed=config.feedback_seed)
        self.generator = torch.Generator(device=model.head.weight.device).manual_seed(config.feedback_seed + 67108879)
        self.observations = 0

    def estimate(self, module, dimension):
        reference = self.model.head.weight
        probes = torch.randn(self.config.mirror_probe_count, dimension, dtype=reference.dtype,
                             device=reference.device, generator=self.generator)
        return estimate_linear_map_from_forward(module, probes, shuffle_responses=False)

    def blend(self, target, candidate):
        candidate = normalize_mirror_map(candidate)
        if self.observations == 0:
            target.copy_(candidate)
        else:
            target.mul_(self.config.mirror_beta).add_(candidate, alpha=1 - self.config.mirror_beta)
            target.copy_(normalize_mirror_map(target))

    @torch.no_grad()
    def update(self):
        for block, maps in zip(self.model.blocks[1:], self.maps[1:]):
            dimension = block.attention.embed_dim
            qkv = self.estimate(block.attention.qkv, dimension)
            for role, targets in (("q", maps.query), ("k", maps.key), ("v", maps.value)):
                for head, target in enumerate(targets):
                    self.blend(target, qkv[block.attention.component_slice(role, head)])
            self.blend(maps.output, self.estimate(block.attention.proj, dimension))
            self.blend(maps.ffn_input, self.estimate(block.ffn.fc1, dimension))
            self.blend(maps.ffn_output, self.estimate(block.ffn.fc2, block.ffn.fc1.out_features))
        self.observations += 1

    def state_dict(self):
        return {"maps": [{f.name: getattr(m, f.name) for f in fields(m)} for m in self.maps],
                "rng": self.generator.get_state(), "observations": self.observations}

    @torch.no_grad()
    def load_state_dict(self, state):
        if len(state["maps"]) != len(self.maps):
            raise ValueError("Mirror checkpoint depth differs")
        for maps, saved in zip(self.maps, state["maps"]):
            for field in fields(maps):
                getattr(maps, field.name).copy_(saved[field.name])
        self.generator.set_state(state["rng"].cpu())
        self.observations = state["observations"]


class ComponentFeedback:
    def __init__(self, model, config):
        self.model, self.config = model, config
        if config.output_probes > model.config.num_classes:
            raise ValueError("Output probes cannot exceed the class count")
        self.mirrors = ForwardMirrorBank(model, config)
        self.generator = torch.Generator(device=model.head.weight.device).manual_seed(config.feedback_seed + 11431)

    @torch.no_grad()
    def prepare(self, caches, error):
        cfg = self.config
        cycles = cfg.mirror_warmup if self.mirrors.observations == 0 else 1
        for _ in range(cycles):
            self.mirrors.update()
        credits, diagnostics = [], {"local_mirror_forward_calls": cycles * 8 * (len(caches) - 1),
                                     "suffix_forward_calls": 0, "suffix_directional_evaluations": 0}
        count = cfg.output_probes or self.model.config.num_classes
        for boundary, cache in enumerate(caches):
            if boundary == len(caches) - 1:
                operator = identity_class_operator(self.model, caches)
            else:
                carrier, dictionary, _ = downstream_component_dictionary(self.model, caches, self.mirrors.maps, boundary)

                def suffix(tokens):
                    for block in self.model.blocks[boundary + 1:]:
                        tokens, _ = block(tokens)
                    return self.model.head(self.model.final_norm(tokens)[:, 0])

                codes = sample_output_codes(carrier, count, self.generator)
                candidate, coefficients, _ = trace_component_operator(
                    suffix, cache.block_output, carrier, dictionary, codes,
                    epsilon=cfg.probe_epsilon, ridge=cfg.operator_ridge,
                    trust_ratio=cfg.trust_ratio, batch_output_queries=True)
                probes = sample_probes(cache.block_output, cfg.validation_probes, self.generator, orthogonal=True)
                response = forward_vector_responses(suffix, cache.block_output, probes, cfg.probe_epsilon)
                operator, accepted = select_operator(carrier, candidate, probes, response)
                diagnostics.update({
                    f"boundary{boundary}_operator_modes": dictionary.shape[1],
                    f"boundary{boundary}_operator_acceptance": float(accepted.float().mean()),
                    f"boundary{boundary}_coefficient_rms": float(coefficients.square().mean().sqrt()),
                    f"boundary{boundary}_correction_ratio": float(((operator-carrier).flatten(1).norm(dim=-1) /
                                                                  carrier.flatten(1).norm(dim=-1).clamp_min(1e-20)).mean()),
                })
                diagnostics["suffix_forward_calls"] += dictionary.shape[1] + 2 * cfg.validation_probes
                diagnostics["suffix_directional_evaluations"] += 2 * (dictionary.shape[1] * count + cfg.validation_probes)
            credits.append(torch.einsum("bc,bcnd->bnd", error, operator))
        return credits, diagnostics

    def state_dict(self):
        return {"mirrors": self.mirrors.state_dict(), "operator_rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.mirrors.load_state_dict(state["mirrors"])
        self.generator.set_state(state["operator_rng"].cpu())
