"""Analytic local VJPs with exact mean-loss scaling and full trainable scope.

Incoming credits already include the loss reduction. No extra B or token
division is applied. A block's input credit is discarded, except at block 0
where it updates the grouped tokenizer. No cross-block reverse pass is used.
"""
import math

import torch
from torch.nn import functional as F

from methods.transformer_ebd.ffn_ebd import gelu_derivative
from methods.transformer_ebd.token_complete_dfa import layer_norm_input_vjp


def assign(parameter, gradient):
    if parameter is not None and parameter.requires_grad:
        parameter.grad = gradient.detach()


def affine_gradients(layer, inputs, modulation):
    assign(layer.weight, modulation.flatten(0, -2).T @ inputs.flatten(0, -2))
    assign(layer.bias, modulation.flatten(0, -2).sum(0))


def normalization_gradients(layer, values, modulation):
    centered = values - values.mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
    axes = tuple(range(values.ndim - 1))
    assign(layer.weight, (modulation * normalized).sum(axes))
    assign(layer.bias, modulation.sum(axes))
    return layer_norm_input_vjp(values, modulation, layer.weight, layer.eps)


@torch.no_grad()
def block_gradients(block, cache, credit):
    r2 = cache.after_attention + cache.ffn_branch
    dr2 = normalization_gradients(block.ln2, r2, credit)
    affine_gradients(block.ffn.fc2, cache.ffn_activation, dr2)
    dz = (dr2 @ block.ffn.fc2.weight) * gelu_derivative(cache.ffn_preactivation)
    affine_gradients(block.ffn.fc1, cache.ffn_input, dz)
    du = dr2 + dz @ block.ffn.fc1.weight
    r1 = cache.block_input + cache.attention_branch
    dr1 = normalization_gradients(block.ln1, r1, du)
    affine_gradients(block.attention.proj, cache.context_concat, dr1)
    batch, tokens, width = dr1.shape
    heads, head_dim = block.attention.num_heads, block.attention.head_dim
    context = (dr1 @ block.attention.proj.weight).reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    raw_score = context @ cache.v.transpose(-1, -2)
    score = cache.attention * (raw_score - (raw_score * cache.attention).sum(-1, keepdim=True))
    query = score @ cache.k / math.sqrt(head_dim)
    key = score.transpose(-1, -2) @ cache.q / math.sqrt(head_dim)
    value = cache.attention.transpose(-1, -2) @ context
    qkv = torch.stack([s.transpose(1, 2) for s in (query, key, value)], dim=2).reshape(batch, tokens, 3 * width)
    affine_gradients(block.attention.qkv, cache.attention_input, qkv)
    return dr1 + qkv @ block.attention.qkv.weight


@torch.no_grad()
def tokenizer_gradients(model, images, credit):
    layer = model.patch_embed
    if layer.groups != 1:
        raise ValueError("Grouped patch convolution is not supported")
    patches = F.unfold(images, kernel_size=layer.kernel_size, dilation=layer.dilation,
                       padding=layer.padding, stride=layer.stride)
    signal = credit[:, 1:].transpose(1, 2)
    assign(layer.weight, torch.einsum("bdn,bkn->dk", signal, patches).reshape_as(layer.weight))
    assign(layer.bias, signal.sum((0, 2)))
    assign(model.cls_token, credit[:, :1].sum(0, keepdim=True))
    assign(model.position, credit.sum(0, keepdim=True))


@torch.no_grad()
def populate_local_gradients(model, images, cache, error, credits):
    if model.norm_layout != "post" or len(credits) != len(model.blocks):
        raise ValueError("Expected one post-LN output credit per block")
    for index, (block, block_cache, credit) in enumerate(zip(model.blocks, cache.block_caches, credits)):
        input_credit = block_gradients(block, block_cache, credit)
        if index == 0:
            tokenizer_gradients(model, images, input_credit)
    affine_gradients(model.head, cache.classifier_input, error)
    final_cls = cache.block_caches[-1].block_output[:, 0]
    normalization_gradients(model.final_norm, final_cls, error @ model.head.weight)
