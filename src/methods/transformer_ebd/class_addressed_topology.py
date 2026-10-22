"""Forward-topology helpers for class-addressed Transformer readouts."""

from __future__ import annotations

import torch

from .configs import ViTConfig


def initialize_classification_tokens(
    position: torch.Tensor,
    classification_tokens: torch.Tensor,
    config: ViTConfig,
    *,
    standard_deviation: float = 0.02,
) -> None:
    """Initialize extra class tokens without changing standard block RNG state."""
    if config.classification_token_count == 1:
        torch.nn.init.trunc_normal_(position, std=standard_deviation)
        torch.nn.init.trunc_normal_(classification_tokens, std=standard_deviation)
        return

    # Consume exactly the draws used by a standard one-CLS ViT. This keeps
    # patch embedding and all block weights matched across token topologies.
    base_position = position.new_empty(1, config.patch_count + 1, config.embed_dim)
    base_cls = classification_tokens.new_empty(1, 1, config.embed_dim)
    torch.nn.init.trunc_normal_(base_position, std=standard_deviation)
    torch.nn.init.trunc_normal_(base_cls, std=standard_deviation)

    code = torch.zeros_like(classification_tokens)
    diagonal = min(config.num_classes, config.embed_dim)
    indices = torch.arange(diagonal, device=classification_tokens.device)
    code[0, indices, indices] = standard_deviation
    with torch.no_grad():
        classification_tokens.copy_(base_cls.expand_as(classification_tokens) + code)
        position[:, : config.num_classes].copy_(
            base_position[:, :1].expand(-1, config.num_classes, -1)
        )
        position[:, config.num_classes :].copy_(base_position[:, 1:])


def block_readout_scale(config: ViTConfig, block_index: int) -> float:
    if not 0 <= block_index < config.depth:
        raise ValueError("block index out of range")
    if config.readout_mode == "class_tokens_final":
        return 1.0 if block_index == config.depth - 1 else 0.0
    if config.readout_mode in ("single_cls_bus", "class_tokens_bus"):
        return 1.0 / float(config.depth) ** 0.5
    return 0.0
