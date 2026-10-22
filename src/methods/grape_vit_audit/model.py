"""Normalization-location control with identical parameter initialization.

GrAPE cites Launay et al.'s post-LN Transformer protocol. This model only
changes LN placement in the existing ViT; it does not reproduce that language
model or add our proposed component feedback. Never use its BlockCache with
the legacy pre-LN EBD hand-derived updates.
"""
import torch
from torch import nn

from methods.transformer_ebd.vit_model import ObservableViT, BlockCache


class PostNormBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        # Reuse every tensor, including initialization. Construct no new weights.
        self.ln1 = original.ln1
        self.attention = original.attention
        self.ln2 = original.ln2
        self.ffn = original.ffn

    def forward(self, tokens, *, average_attention_heads=False, ablated_attention_heads=()):
        attention_branch, attention_cache = self.attention(
            tokens, average_heads=average_attention_heads,
            ablated_heads=ablated_attention_heads)
        q, k, v, scores, attention, context, context_concat = attention_cache
        after_attention = self.ln1(tokens + attention_branch)
        ffn_branch, ffn_preactivation, ffn_activation = self.ffn(after_attention)
        output = self.ln2(after_attention + ffn_branch)
        return output, BlockCache(
            block_input=tokens, attention_input=tokens, q=q, k=k, v=v,
            scores=scores, attention=attention, context=context,
            context_concat=context_concat, attention_branch=attention_branch,
            after_attention=after_attention, ffn_input=after_attention,
            ffn_preactivation=ffn_preactivation, ffn_activation=ffn_activation,
            ffn_branch=ffn_branch, block_output=output)


class AuditedViT(ObservableViT):
    def __init__(self, config, norm_layout="pre"):
        super().__init__(config)
        if norm_layout not in ("pre", "post"):
            raise ValueError("norm_layout must be pre or post")
        self.norm_layout = norm_layout
        if norm_layout == "post":
            self.blocks = nn.ModuleList(PostNormBlock(block) for block in self.blocks)
        self.activation_diagnostics = {}

    def forward_with_cache(self, images, **kwargs):
        cache = super().forward_with_cache(images, **kwargs)
        # Measure full-forward activations, without retaining a reverse graph.
        # No diagnostic BP or gradient clipping enters training.
        with torch.no_grad():
            self.activation_diagnostics = {
                f"block{i}_output_rms": float(c.block_output.square().mean().sqrt())
                for i, c in enumerate(cache.block_caches)}
            self.activation_diagnostics.update({
                f"block{i}_ffn_preactivation_rms": float(c.ffn_preactivation.square().mean().sqrt())
                for i, c in enumerate(cache.block_caches)})
        return cache
