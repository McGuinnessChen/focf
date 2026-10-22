"""Vision transformer configuration."""


from __future__ import annotations


from dataclasses import asdict, dataclass


from typing import Dict


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    channels: int = 3
    num_classes: int = 10
    embed_dim: int = 128
    depth: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    layer_norm_eps: float = 1e-6
    freeze_position: bool = True
    freeze_layer_norm: bool = True
    freeze_patch_embed_manual: bool = True
    readout_mode: str = "standard"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if min(self.depth, self.num_heads, self.ffn_dim) <= 0:
            raise ValueError("depth, num_heads, and ffn_dim must be positive")
        if self.readout_mode not in (
            "standard",
            "single_cls_bus",
            "class_tokens_final",
            "class_tokens_bus",
        ):
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")

    @property
    def patch_count(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def classification_token_count(self) -> int:
        return self.num_classes if self.readout_mode.startswith("class_tokens") else 1

    @property
    def token_count(self) -> int:
        return self.patch_count + self.classification_token_count

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
