"""Random-dictionary control with the same fitting and acceptance budget."""


from __future__ import annotations


import torch


from methods.component_postln.feedback import ForwardMirrorBank


from methods.transformer_ebd.component_route_operator import sample_output_codes, select_operator, trace_component_operator


from methods.transformer_ebd.forward_response_credit import forward_vector_responses, sample_probes


from methods.component_postln.operators import downstream_component_dictionary


class RandomRouteFeedback:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.mirrors = ForwardMirrorBank(model, config)
        self.generator = torch.Generator(device=model.head.weight.device).manual_seed(config.feedback_seed + 11431)

    @torch.no_grad()
    def prepare(self, caches, error):
        cfg = self.config
        cycles = cfg.mirror_warmup if self.mirrors.observations == 0 else 1
        for _ in range(cycles):
            self.mirrors.update()
        count = cfg.output_probes or self.model.config.num_classes
        credits, diagnostics = [], {
            "local_mirror_forward_calls": cycles * 8 * (len(caches) - 1),
            "suffix_forward_calls": 0, "suffix_directional_evaluations": 0,
        }
        for boundary, cache in enumerate(caches):
            if boundary == len(caches) - 1:
                carrier = __import__("methods.transformer_ebd.component_route_operator", fromlist=["identity_class_operator"]).identity_class_operator(self.model, caches)
                operator = carrier
            else:
                carrier, structured, _ = downstream_component_dictionary(
                    self.model, caches, self.mirrors.maps, boundary)
                random = torch.randn(structured.shape, device=structured.device,
                                     dtype=structured.dtype, generator=self.generator)
                norms = random.flatten(2).norm(dim=-1)
                carrier_norm = carrier.flatten(1).norm(dim=-1)
                dictionary = random * (carrier_norm[:, None] / norms.clamp_min(1e-20))[:, :, None, None, None]

                def suffix(tokens, boundary=boundary):
                    for block in self.model.blocks[boundary + 1:]:
                        tokens, _ = block(tokens)
                    return self.model.head(self.model.final_norm(tokens)[:, 0])

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
                    f"boundary{boundary}_operator_modes": dictionary.shape[1],
                    f"boundary{boundary}_operator_acceptance": float(accepted.float().mean()),
                    f"boundary{boundary}_coefficient_rms": float(coefficients.square().mean().sqrt()),
                })
                diagnostics["suffix_forward_calls"] += dictionary.shape[1] + 2 * cfg.validation_probes
                diagnostics["suffix_directional_evaluations"] += 2 * (dictionary.shape[1] * count + cfg.validation_probes)
            credits.append(torch.einsum("bc,bcnd->bnd", error, operator))
        return credits, diagnostics

    def state_dict(self):
        return {"mirrors": self.mirrors.state_dict(), "operator_rng": self.generator.get_state()}
