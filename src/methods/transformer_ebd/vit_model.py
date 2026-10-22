"""A small Pre-LN ViT whose forward pass exposes every local EBD variable."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .configs import ViTConfig
from .class_addressed_topology import (
    block_readout_scale,
    initialize_classification_tokens,
)


@dataclass
class BlockCache:
    block_input: torch.Tensor
    attention_input: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scores: torch.Tensor
    attention: torch.Tensor
    context: torch.Tensor
    context_concat: torch.Tensor
    attention_branch: torch.Tensor
    after_attention: torch.Tensor
    ffn_input: torch.Tensor
    ffn_preactivation: torch.Tensor
    ffn_activation: torch.Tensor
    ffn_branch: torch.Tensor
    block_output: torch.Tensor
    readout_input: Optional[torch.Tensor] = None
    logit_contribution: Optional[torch.Tensor] = None


@dataclass
class ViTForwardCache:
    patch_tokens: torch.Tensor
    block_caches: List[BlockCache]
    final_tokens: torch.Tensor
    classifier_input: torch.Tensor
    logits: torch.Tensor
    block_logit_contributions: List[torch.Tensor]


class ObservableAttention(nn.Module):
    """Combined-QKV attention with an explicit, testable row layout."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def component_slice(self, component: str, head: Optional[int] = None) -> slice:
        offsets = {"q": 0, "k": self.embed_dim, "v": 2 * self.embed_dim}
        if component not in offsets:
            raise ValueError(f"Unknown QKV component: {component}")
        start = offsets[component]
        width = self.embed_dim
        if head is not None:
            if not 0 <= head < self.num_heads:
                raise ValueError("head index out of range")
            start += head * self.head_dim
            width = self.head_dim
        return slice(start, start + width)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        average_heads: bool = False,
        ablated_heads: Sequence[int] = (),
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, token_count, _ = tokens.shape
        combined = self.qkv(tokens)
        combined = combined.view(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        q, k, v = combined.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        if average_heads:
            attention = attention.mean(dim=1, keepdim=True).expand_as(
                attention
            )
        context = torch.matmul(attention, v)
        if ablated_heads:
            context = context.clone()
            # PyTorch 1.12 CUDA has an internal assertion for the equivalent
            # tuple-based advanced assignment; basic slicing is deterministic.
            for head in ablated_heads:
                context[:, int(head)] = 0
        context_concat = context.transpose(1, 2).reshape(
            batch_size, token_count, self.embed_dim
        )
        branch = self.proj(context_concat)
        return branch, (q, k, v, scores, attention, context, context_concat)


class ObservableFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preactivation = self.fc1(tokens)
        activation = F.gelu(preactivation, approximate="none")
        branch = self.fc2(activation)
        return branch, preactivation, activation


class ObservableTransformerBlock(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attention = ObservableAttention(config.embed_dim, config.num_heads)
        self.ln2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.ffn = ObservableFFN(config.embed_dim, config.ffn_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        average_attention_heads: bool = False,
        ablated_attention_heads: Sequence[int] = (),
    ) -> Tuple[torch.Tensor, BlockCache]:
        attention_input = self.ln1(tokens)
        attention_branch, attention_cache = self.attention(
            attention_input,
            average_heads=average_attention_heads,
            ablated_heads=ablated_attention_heads,
        )
        q, k, v, scores, attention, context, context_concat = attention_cache
        after_attention = tokens + attention_branch
        ffn_input = self.ln2(after_attention)
        ffn_branch, ffn_preactivation, ffn_activation = self.ffn(ffn_input)
        output = after_attention + ffn_branch
        return output, BlockCache(
            block_input=tokens,
            attention_input=attention_input,
            q=q,
            k=k,
            v=v,
            scores=scores,
            attention=attention,
            context=context,
            context_concat=context_concat,
            attention_branch=attention_branch,
            after_attention=after_attention,
            ffn_input=ffn_input,
            ffn_preactivation=ffn_preactivation,
            ffn_activation=ffn_activation,
            ffn_branch=ffn_branch,
            block_output=output,
        )


class ObservableViT(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                config.classification_token_count,
                config.embed_dim,
            )
        )
        self.position = nn.Parameter(
            torch.zeros(1, config.token_count, config.embed_dim),
            requires_grad=not config.freeze_position,
        )
        self.blocks = nn.ModuleList(
            ObservableTransformerBlock(config) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.embed_dim, config.num_classes, bias=True)
        self.block_readouts = nn.ModuleList()
        construction_rng_state = torch.random.get_rng_state()
        self.class_readout: Optional[nn.Linear] = None
        self.class_bias: Optional[nn.Parameter] = None
        if config.readout_mode == "single_cls_bus":
            self.block_readouts = nn.ModuleList(
                nn.Linear(config.embed_dim, config.num_classes, bias=True)
                for _ in range(config.depth)
            )
        elif config.readout_mode.startswith("class_tokens"):
            # Token index supplies class identity; one feature readout is shared.
            self.class_readout = nn.Linear(config.embed_dim, 1, bias=False)
            self.class_bias = nn.Parameter(torch.zeros(config.num_classes))
        torch.random.set_rng_state(construction_rng_state)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_classification_tokens(self.position, self.cls_token, self.config)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attention.qkv.weight)
            nn.init.xavier_uniform_(block.attention.proj.weight)
            nn.init.xavier_uniform_(block.ffn.fc1.weight)
            nn.init.zeros_(block.ffn.fc1.bias)
            nn.init.xavier_uniform_(block.ffn.fc2.weight)
            nn.init.zeros_(block.ffn.fc2.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for readout in self.block_readouts:
            nn.init.xavier_uniform_(readout.weight)
            nn.init.zeros_(readout.bias)
        if self.class_readout is not None:
            nn.init.xavier_uniform_(self.class_readout.weight)
            nn.init.zeros_(self.class_bias)

    def tokenize(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1) + self.position

    def _direct_readout(
        self, block_index: int, tokens: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        prefix = self.config.classification_token_count
        readout_input = self.final_norm(tokens[:, :prefix])
        scale = block_readout_scale(self.config, block_index)
        if self.config.readout_mode == "standard":
            contribution = tokens.new_zeros(tokens.shape[0], self.config.num_classes)
        elif self.config.readout_mode == "single_cls_bus":
            contribution = scale * self.block_readouts[block_index](readout_input[:, 0])
        else:
            if self.class_readout is None:
                raise RuntimeError("Class-token readout is not initialized")
            contribution = (
                scale * self.class_readout(readout_input).squeeze(-1)
            )
        return readout_input, contribution

    def forward_with_cache(
        self,
        images: torch.Tensor,
        *,
        ablated_blocks: Sequence[int] = (),
        averaged_attention_blocks: Sequence[int] = (),
        ablated_attention_heads: Mapping[int, Sequence[int]] | None = None,
        disabled_readout_blocks: Sequence[int] = (),
    ) -> ViTForwardCache:
        tokens = self.tokenize(images)
        patch_tokens = tokens
        caches: List[BlockCache] = []
        contributions: List[torch.Tensor] = []
        ablated = set(int(index) for index in ablated_blocks)
        disabled_readouts = set(int(index) for index in disabled_readout_blocks)
        averaged = set(int(index) for index in averaged_attention_blocks)
        head_ablations = {} if ablated_attention_heads is None else ablated_attention_heads
        for index, block in enumerate(self.blocks):
            output, cache = block(
                tokens,
                average_attention_heads=index in averaged,
                ablated_attention_heads=head_ablations.get(index, ()),
            )
            tokens = tokens if index in ablated else output
            readout_input, contribution = self._direct_readout(index, tokens)
            if index in ablated or index in disabled_readouts:
                contribution = torch.zeros_like(contribution)
            cache.readout_input = readout_input
            cache.logit_contribution = contribution
            caches.append(cache)
            contributions.append(contribution)
        final_tokens = self.final_norm(tokens)
        prefix = self.config.classification_token_count
        classifier_input = final_tokens[:, :prefix].mean(dim=1)
        if self.config.readout_mode == "standard":
            logits = self.head(classifier_input)
        else:
            logits = torch.stack(contributions, dim=0).sum(dim=0)
            if self.class_bias is not None:
                logits = logits + self.class_bias
        return ViTForwardCache(
            patch_tokens=patch_tokens,
            block_caches=caches,
            final_tokens=final_tokens,
            classifier_input=classifier_input,
            logits=logits,
            block_logit_contributions=contributions,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(images).logits

    def freeze_for_manual_learning(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def enable_bp_learning(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if self.config.freeze_position:
            self.position.requires_grad_(False)
        if self.config.freeze_layer_norm:
            for module in list(self.blocks) + [self.final_norm]:
                for name, parameter in module.named_parameters():
                    if "ln" in name or module is self.final_norm:
                        parameter.requires_grad_(False)

    def enable_full_bp_learning(self) -> None:
        """Enable the conventional end-to-end BP reference parameter set."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def enable_protocol_matched_bp_learning(self) -> None:
        """Match the parameter set updated by the manual broadcast methods."""
        self.freeze_for_manual_learning()
        for block in self.blocks:
            # Manual EBD/DFA update Q/K/V/O and both FFN affine maps, but not
            # patchification, token embeddings, positional state, or LayerNorm.
            for parameter in block.attention.parameters():
                parameter.requires_grad_(True)
            for parameter in block.ffn.parameters():
                parameter.requires_grad_(True)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)

    def qkv_slices(self, block_index: int) -> dict:
        attention = self.blocks[block_index].attention
        return {
            component: attention.component_slice(component)
            for component in ("q", "k", "v")
        }

    def clone_qk_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        snapshots = []
        for block in self.blocks:
            weight = block.attention.qkv.weight
            snapshots.append(
                (
                    weight[block.attention.component_slice("q")].detach().clone(),
                    weight[block.attention.component_slice("k")].detach().clone(),
                )
            )
        return snapshots

    def clone_representation_weights(self) -> List[Tuple[torch.Tensor, ...]]:
        """Snapshot V/O/FFN parameters without including routing or the head."""
        snapshots = []
        for block in self.blocks:
            v_slice = block.attention.component_slice("v")
            snapshots.append(
                (
                    block.attention.qkv.weight[v_slice].detach().clone(),
                    block.attention.proj.weight.detach().clone(),
                    block.ffn.fc1.weight.detach().clone(),
                    block.ffn.fc1.bias.detach().clone(),
                    block.ffn.fc2.weight.detach().clone(),
                    block.ffn.fc2.bias.detach().clone(),
                )
            )
        return snapshots
