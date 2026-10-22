"""Initialization of the channel maps used by FOCF."""


from __future__ import annotations


from dataclasses import dataclass


from typing import List, Tuple


import torch


from .vit_model import ObservableViT


@dataclass
class ResidualPathMaps:
    linear: torch.Tensor
    output: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    ffn_output: torch.Tensor
    ffn_input: torch.Tensor

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.linear,
            self.output,
            self.query,
            self.key,
            self.value,
            self.ffn_output,
            self.ffn_input,
        )


def _orthogonal_map(
    rows: int,
    columns: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    matrix = torch.randn(
        rows, columns, generator=generator, dtype=dtype, device=device
    )
    if rows <= columns:
        return torch.linalg.qr(
            matrix.transpose(0, 1), mode="reduced"
        ).Q.transpose(0, 1)
    return torch.linalg.qr(matrix, mode="reduced").Q


def make_residual_path_maps(
    model: ObservableViT, *, seed: int
) -> List[ResidualPathMaps]:
    """Construct a fixed feedback Transformer independent of forward weights."""
    reference = model.head.weight
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed) + 90909191)
    config = model.config
    result = []
    for _ in range(config.depth):
        result.append(
            ResidualPathMaps(
                linear=_orthogonal_map(
                    config.embed_dim,
                    config.embed_dim,
                    generator=generator,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                output=_orthogonal_map(
                    config.embed_dim,
                    config.embed_dim,
                    generator=generator,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                query=torch.stack(
                    [
                        _orthogonal_map(
                            config.head_dim,
                            config.embed_dim,
                            generator=generator,
                            dtype=reference.dtype,
                            device=reference.device,
                        )
                        for _ in range(config.num_heads)
                    ]
                ),
                key=torch.stack(
                    [
                        _orthogonal_map(
                            config.head_dim,
                            config.embed_dim,
                            generator=generator,
                            dtype=reference.dtype,
                            device=reference.device,
                        )
                        for _ in range(config.num_heads)
                    ]
                ),
                value=torch.stack(
                    [
                        _orthogonal_map(
                            config.head_dim,
                            config.embed_dim,
                            generator=generator,
                            dtype=reference.dtype,
                            device=reference.device,
                        )
                        for _ in range(config.num_heads)
                    ]
                ),
                ffn_output=_orthogonal_map(
                    config.embed_dim,
                    config.ffn_dim,
                    generator=generator,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
                ffn_input=_orthogonal_map(
                    config.ffn_dim,
                    config.embed_dim,
                    generator=generator,
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            )
        )
    return result
