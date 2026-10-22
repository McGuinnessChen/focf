"""Mathematical and information-flow checks, not accuracy experiments."""
import pytest
import torch

from methods.grape.core import (
    batch_rank_one_estimate, calibration_due, column_cosine_refinement,
    forward_jvp, project_error,
)
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.vit_model import ObservableViT


def test_refinement_matches_autograd_of_stated_objective():
    torch.manual_seed(261011)
    a = torch.randn(4, 13, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(a)
    target[:, 2] = 0  # unobservable coordinate must not generate a NaN
    eps, lr = 1e-12, 0.7
    objective = 1 - (
        a / a.norm(dim=0, keepdim=True).clamp_min(eps)
        * target / target.norm(dim=0, keepdim=True).clamp_min(eps)
    ).sum(0).mean()
    gradient, = torch.autograd.grad(objective, a)
    expected = a.detach() - lr * gradient
    expected /= expected.norm(dim=0, keepdim=True) + eps
    actual = column_cosine_refinement(a, target, learning_rate=lr, epsilon=eps)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.norm(dim=0), torch.ones(13, dtype=a.dtype))


def test_full_isotropic_design_recovers_linear_jacobian_and_its_adjoint():
    torch.manual_seed(261012)
    n, c = 7, 3
    jacobian = torch.randn(c, n, dtype=torch.float64)
    probes = n ** 0.5 * torch.eye(n, dtype=torch.float64)
    _, responses = forward_jvp(lambda h: h @ jacobian.T, torch.zeros_like(probes), probes)
    estimate = batch_rank_one_estimate(probes, responses)
    torch.testing.assert_close(estimate, jacobian, atol=1e-12, rtol=1e-12)
    errors = torch.randn(n, c, dtype=torch.float64)
    credit = project_error(estimate, errors)
    torch.testing.assert_close((credit * probes).sum(1), (errors * responses).sum(1))


def test_core_does_not_call_reverse_mode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reverse-mode used inside GrAPE estimator/refinement")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    h, p = torch.randn(5, 6), torch.randn(5, 6)
    w = torch.randn(6, 3, requires_grad=True)
    logits, response = forward_jvp(lambda z: torch.tanh(z @ w), h, p)
    estimate = batch_rank_one_estimate(p, response)
    a = column_cosine_refinement(torch.randn_like(estimate), estimate, learning_rate=1.)
    credit = project_error(a, torch.ones_like(logits))
    assert torch.isfinite(credit).all()
    assert not any(x.requires_grad for x in [logits, response, estimate, a, credit])
    assert w.grad is None


def test_vit_suffix_jvp_matches_independent_reverse_jacobian():
    torch.manual_seed(261013)
    model = ObservableViT(ViTConfig(
        image_size=8, patch_size=4, embed_dim=4, num_heads=2,
        ffn_dim=6, depth=2, num_classes=3,
    )).double()
    h = torch.randn(1, 5, 4, dtype=torch.float64)
    probe = torch.randn_like(h)
    def suffix(z):
        out, _ = model.blocks[1](z)
        return model.head(model.final_norm(out)[:, 0])
    logits, actual = forward_jvp(suffix, h, probe)
    exact = torch.autograd.functional.jacobian(suffix, h).reshape(3, -1)
    expected = (exact @ probe.flatten()).unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    step = 1e-5
    finite = (suffix(h + step * probe) - suffix(h - step * probe)) / (2 * step)
    torch.testing.assert_close(actual, finite, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(logits, suffix(h))


def test_batch_estimate_is_mean_of_outer_products_not_product_of_means():
    p = torch.tensor([[1., 0.], [-1., 0.]])
    g = torch.tensor([[2., 3.], [-2., -3.]])
    actual = batch_rank_one_estimate(p, g)
    torch.testing.assert_close(actual, torch.tensor([[2., 0.], [3., 0.]]))
    assert not torch.equal(actual, g.mean(0)[:, None] @ p.mean(0)[None, :])


def test_calibration_count_is_one_batch_per_interval_not_one_epoch_of_bp():
    assert sum(calibration_due(e, 0) for e in range(1, 101)) == 0
    assert sum(calibration_due(e, 10) for e in range(1, 101)) == 10
    assert sum(calibration_due(e, 1) for e in range(1, 101)) == 100
    with pytest.raises(ValueError):
        calibration_due(1, -1)
