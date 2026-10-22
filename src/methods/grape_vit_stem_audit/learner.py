"""Train the input embedding through the first block's local feedback graph.

The Transformer reference cited by GrAPE defaults to no separate embedding
feedback hook. This is a grouping control for our ViT adaptation, not a claim
about an unpublished official ViT implementation. Cross-block reverse-mode
credit propagation remains cut at each encoder output.
"""
import torch

from methods.grape_vit_audit.learner import AuditedGrAPELearner


class GroupedStemGrAPELearner(AuditedGrAPELearner):
    def __init__(self, model, config):
        super().__init__(model, config)
        if config.parameter_scope == "all":
            # Use the same block feedback initialization and probe stream as
            # blocks_head. Only the local graph's trainable scope is enlarged.
            self.boundary_indices = list(range(model.config.depth))
            self.generator.manual_seed(config.feedback_seed)
            reference = model.head.weight
            self.feedback = []
            for _ in self.boundary_indices:
                a = torch.randn(model.config.num_classes,
                                model.config.token_count * model.config.embed_dim,
                                generator=self.generator, device=reference.device,
                                dtype=reference.dtype)
                a /= a.norm(dim=0, keepdim=True) + 1e-12
                self.feedback.append(a)
            self.initial_feedback = [a.clone() for a in self.feedback]

    def populate_local_gradients(self, images, cache, error, credits):
        if self.config.parameter_scope != "all":
            return super().populate_local_gradients(images, cache, error, credits)
        for index, block in enumerate(self.model.blocks):
            if index == 0:
                tokens = self.model.tokenize(images.detach())
                parameters = (list(self.model.patch_embed.parameters()) +
                              [self.model.cls_token, self.model.position] +
                              list(block.parameters()))
            else:
                tokens = cache.block_caches[index].block_input.detach()
                parameters = list(block.parameters())
            output, _ = block(tokens)
            self.local_parameter_gradients(output, parameters, credits[index])
        last = cache.block_caches[-1].block_output.detach()
        logits = self.model.head(self.model.final_norm(last)[:, 0])
        self.local_parameter_gradients(
            logits, list(self.model.final_norm.parameters()) + list(self.model.head.parameters()), error)
