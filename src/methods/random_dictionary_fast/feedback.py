"""Keep B = M + sum_r a_r T_r and every random draw/query unchanged.

The old implementation constructs a structured dictionary only to obtain its
shape. Its values never enter this control. Count those modes directly, reuse
the common carrier, and normalize the random allocation in place. Optional
CLS suffix evaluation omits only outputs unused by the classifier; its smaller
matrix products require a float32 continuation audit before formal use.
"""
import torch

from methods.cifar10_random_dictionary.trainer import ResumableRandomRouteFeedback
from methods.component_scaling_fast.feedback import cls_suffix
from methods.transformer_ebd.component_route_operator import (
    identity_class_operator, sample_output_codes, select_operator,
    trace_component_operator,
)
from methods.transformer_ebd.forward_response_credit import (
    forward_vector_responses, sample_probes,
)


class FastRandomRouteFeedback(ResumableRandomRouteFeedback):
    def __init__(self, model, config, *, cls_only=False):
        super().__init__(model, config)
        self.cls_only = cls_only

    @torch.no_grad()
    def prepare(self, caches, error):
        cfg = self.config
        cycles = cfg.mirror_warmup if self.mirrors.observations == 0 else 1
        # Preserve mirror state/RNG even though random directions do not use it.
        for _ in range(cycles):
            self.mirrors.update()
        count = cfg.output_probes or self.model.config.num_classes
        carrier = identity_class_operator(self.model, caches)
        carrier_norm = carrier.flatten(1).norm(dim=-1)
        credits, diagnostics = [], {
            "local_mirror_forward_calls": cycles * 8 * (len(caches) - 1),
            "suffix_forward_calls": 0, "suffix_directional_evaluations": 0,
        }
        for boundary, cache in enumerate(caches):
            if boundary == len(caches) - 1:
                operator = carrier
            else:
                modes = 1 + sum(2 + 3 * block.attention.num_heads
                                for block in self.model.blocks[boundary + 1:])
                shape = (carrier.shape[0], modes, *carrier.shape[1:])
                dictionary = torch.randn(shape, device=carrier.device,
                                         dtype=carrier.dtype, generator=self.generator)
                norms = dictionary.flatten(2).norm(dim=-1)
                dictionary.mul_((carrier_norm[:, None] /
                                 norms.clamp_min(1e-20))[:, :, None, None, None])

                def suffix(tokens, boundary=boundary):
                    if self.cls_only:
                        return cls_suffix(self.model, tokens, boundary)
                    for block in self.model.blocks[boundary + 1:]:
                        tokens = block(tokens)[0]
                    return self.model.head(self.model.final_norm(tokens)[:, 0])

                # Preserve boundary order, tensor shapes and shared RNG stream.
                codes = sample_output_codes(carrier, count, self.generator)
                candidate, coefficients, _ = trace_component_operator(
                    suffix, cache.block_output, carrier, dictionary, codes,
                    epsilon=cfg.probe_epsilon, ridge=cfg.operator_ridge,
                    trust_ratio=cfg.trust_ratio, batch_output_queries=True)
                probes = sample_probes(cache.block_output, cfg.validation_probes,
                                       self.generator, orthogonal=True)
                responses = forward_vector_responses(suffix, cache.block_output, probes, cfg.probe_epsilon)
                operator, accepted = select_operator(carrier, candidate, probes, responses)
                diagnostics.update({
                    f"boundary{boundary}_operator_modes": modes,
                    f"boundary{boundary}_operator_acceptance": float(accepted.float().mean()),
                    f"boundary{boundary}_coefficient_rms": float(coefficients.square().mean().sqrt()),
                })
                diagnostics["suffix_forward_calls"] += modes + 2 * cfg.validation_probes
                diagnostics["suffix_directional_evaluations"] += 2 * (modes * count + cfg.validation_probes)
                del dictionary, candidate, coefficients, probes, responses, codes
            credits.append(torch.einsum("bc,bcnd->bnd", error, operator))
        return credits, diagnostics
