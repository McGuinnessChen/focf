import copy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from methods.component_postln.feedback import ForwardMirrorBank
from methods.component_postln.learner import CommonConfig, CommonLearner
from methods.component_postln.local_gradients import populate_local_gradients
from methods.component_postln.operators import block_component_operators
from methods.grape.learner import GrAPEConfig, objective, output_derivative
from methods.grape_vit_audit.model import AuditedViT
from methods.grape_vit_stem_audit.learner import GroupedStemGrAPELearner
from methods.transformer_ebd.configs import ViTConfig


def model():
    torch.manual_seed(11901)
    net = AuditedViT(ViTConfig(image_size=8, patch_size=4, embed_dim=8,
                              depth=3, num_heads=2, ffn_dim=12, num_classes=3), "post").double()
    # Nonuniform learned affine parameters expose LN ordering/scaling mistakes.
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, torch.nn.LayerNorm):
                layer.weight.uniform_(0.5, 1.5)
                layer.bias.uniform_(-0.2, 0.2)
    return net


def oracle_maps(block):
    attention = block.attention
    weights = attention.qkv.weight.detach()
    return SimpleNamespace(output=attention.proj.weight.detach(),
        query=weights[attention.component_slice("q")].reshape(attention.num_heads, attention.head_dim, -1),
        key=weights[attention.component_slice("k")].reshape(attention.num_heads, attention.head_dim, -1),
        value=weights[attention.component_slice("v")].reshape(attention.num_heads, attention.head_dim, -1),
        ffn_input=block.ffn.fc1.weight.detach(), ffn_output=block.ffn.fc2.weight.detach())


def test_postln_components_reconstruct_exact_block_adjoint():
    block = model().blocks[1]
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    y, cache = block(x)
    carrier = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    parts, names = block_component_operators(carrier, cache, block, oracle_maps(block))
    exact = torch.stack([torch.autograd.grad(y, x, carrier[:, c], retain_graph=True)[0] for c in range(3)], dim=1)
    torch.testing.assert_close(carrier + parts.sum(1), exact, atol=1e-11, rtol=1e-10)
    assert names[:2] == ["normalization", "ffn"]
    assert parts[:, 0].norm() > 1e-3


def test_postln_qk_paths_vanish_with_constant_values():
    block = model().blocks[1]
    _, cache = block(torch.randn(2, 5, 8, dtype=torch.float64))
    cache = replace(cache, v=cache.v[:, :, :1].expand_as(cache.v))
    parts, names = block_component_operators(torch.randn(2, 3, 5, 8, dtype=torch.float64), cache, block, oracle_maps(block))
    for i, name in enumerate(names):
        if name.endswith(("query", "key")):
            assert parts[:, i].abs().max() < 1e-11
        elif name.endswith("value"):
            assert parts[:, i].norm() > 1e-4


@pytest.mark.parametrize("loss_name", ["cross_entropy", "mse"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_analytic_local_gradients_recover_all_bp_parameters_with_exact_credits(loss_name, batch_size):
    net = model()
    net.enable_full_bp_learning()
    x = torch.randn(batch_size, 3, 8, 8, dtype=torch.float64)
    y = torch.arange(batch_size) % 3
    cache = net.forward_with_cache(x)
    loss = objective(cache.logits, y, loss_name)
    credits = torch.autograd.grad(loss, [c.block_output for c in cache.block_caches], retain_graph=True)
    expected = torch.autograd.grad(loss, list(net.parameters()))
    populate_local_gradients(net, x, cache, output_derivative(cache.logits, y, loss_name), credits)
    for (name, p), grad in zip(net.named_parameters(), expected):
        torch.testing.assert_close(p.grad, grad, atol=2e-11, rtol=1e-9, msg=name)


def test_arbitrary_feedback_matches_isolated_autograd_without_cross_block_credit():
    net = model()
    oracle = GroupedStemGrAPELearner(copy.deepcopy(net), GrAPEConfig(parameter_scope="all"))
    net.enable_full_bp_learning()
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        cache = net.forward_with_cache(x)
        error = torch.randn_like(cache.logits)
        credits = [torch.randn_like(c.block_output) for c in cache.block_caches]
    populate_local_gradients(net, x, cache, error, credits)
    oracle.populate_local_gradients(x, cache, error, credits)
    for (name, p), q in zip(net.named_parameters(), oracle.model.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-10, rtol=1e-9, msg=name)


@pytest.mark.parametrize("method", ["component", "identity"])
def test_training_forbids_reverse_autograd_and_calibration(method, monkeypatch):
    learner = CommonLearner(model(), CommonConfig(method=method, mirror_warmup=2, validation_probes=2))
    before = {n: p.clone() for n, p in learner.model.named_parameters()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Reverse autograd called")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    x, y = torch.randn(2, 3, 8, 8, dtype=torch.float64), torch.tensor([0, 2])
    logits, _, _ = learner.step(x, y)
    assert torch.isfinite(logits).all()
    for name, p in learner.model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert not torch.equal(p, before[name]), name
    assert learner.calibration_batches == 0
    with pytest.raises(ValueError, match="prohibit"):
        learner.calibrate(x, y)




@pytest.mark.parametrize("method", ["grape", "dfa", "bp"])
def test_common_baselines_preserve_prior_implementation_bitwise(method):
    cfg = CommonConfig(method=method)
    old_cfg = GrAPEConfig(**{f.name: getattr(cfg, f.name) for f in fields(GrAPEConfig)})
    common = CommonLearner(model(), cfg)
    old = GroupedStemGrAPELearner(model(), old_cfg)
    x, y = torch.randn(2, 3, 8, 8, dtype=torch.float64), torch.tensor([0, 2])
    for _ in range(2):
        a, b = common.step(x, y), old.step(x, y)
        assert torch.equal(a[0], b[0]) and a[1:] == b[1:]
        for p, q in zip(common.model.parameters(), old.model.parameters()):
            assert torch.equal(p, q)


def test_checkpoint_preserves_mirrors_probe_stream_and_adam_bitwise():
    config = CommonConfig(method="component", mirror_warmup=2, validation_probes=2, output_probes=2)
    original = CommonLearner(model(), config)
    x, y = torch.randn(2, 3, 8, 8, dtype=torch.float64), torch.tensor([0, 2])
    original.step(x, y)
    restored = CommonLearner(model(), config)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    a, b = original.step(x, y), restored.step(x, y)
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]
    for p, q in zip(original.model.parameters(), restored.model.parameters()):
        assert torch.equal(p, q)
    assert original.component.mirrors.observations == 3
    assert torch.equal(original.component.generator.get_state(), restored.component.generator.get_state())
