import copy

import pytest
import torch

from methods.component_scaling.learner import ScalingConfig, ScalingLearner
from methods.component_scaling_fast.learner import CachedScalingLearner
from methods.component_scaling_fast.feedback import cls_suffix
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.configs import ViTConfig


def model(classes=12, depth=3):
    torch.manual_seed(12012)
    return AuditedViT(ViTConfig(image_size=8, patch_size=4, embed_dim=8, num_heads=2,
                                depth=depth, ffn_dim=12, num_classes=classes), "post").double()


def assert_state_close(a, b):
    if isinstance(a, torch.Tensor):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-7)
        else:
            assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_close(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("classes,count,depth", [(5, 0, 3), (100, 3, 3), (200, 3, 4), (5, 3, 1)])
@pytest.mark.parametrize("cls_only", [False, True])
def test_reuse_preserves_feedback_updates_rng_and_checkpoint(classes, count, depth, cls_only, monkeypatch):
    cfg = ScalingConfig(method="component", output_probes=count, mirror_warmup=2,
                        validation_probes=2, feedback_sample_chunk=2, output_query_chunk=2)
    reference = ScalingLearner(model(classes, depth), cfg)
    actual = CachedScalingLearner(model(classes, depth), cfg, cls_only=cls_only)
    x, y = torch.randn(5, 3, 8, 8, dtype=torch.float64), torch.arange(5) % classes
    def forbidden(*args, **kwargs):
        raise AssertionError("Optimization must not introduce reverse autograd")
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    # Both first-minibatch mirror warmup and later persistent mirror updates.
    for _ in range(2):
        a, b = actual.step(x, y), reference.step(x, y)
        torch.testing.assert_close(a[0], b[0], atol=2e-10, rtol=2e-9)
        for (name, p), q in zip(actual.model.named_parameters(), reference.model.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-9, rtol=2e-7, msg=name)
        for key in b[2]:
            assert a[2][key] == pytest.approx(b[2][key], abs=2e-9, rel=2e-7), key
        assert_state_close(actual.state_dict(), reference.state_dict())
        assert actual.calibration_batches == reference.calibration_batches == 0
    # Resume FROM the frozen implementation, including Adam and both RNGs.
    restored = CachedScalingLearner(model(classes, depth), cfg, cls_only=cls_only)
    restored.load_state_dict(copy.deepcopy(reference.state_dict()))
    reference.step(x, y)
    restored.step(x, y)
    assert_state_close(restored.state_dict(), reference.state_dict())


def test_last_block_unused_outputs_can_be_removed():
    net = model()
    with torch.no_grad():
        cache = net.forward_with_cache(torch.randn(5, 3, 8, 8, dtype=torch.float64))
        for boundary in range(2):
            tokens = cache.block_caches[boundary].block_output
            expected = tokens
            for block in net.blocks[boundary + 1:]:
                expected, _ = block(expected)
            expected = net.head(net.final_norm(expected)[:, 0])
            torch.testing.assert_close(cls_suffix(net, tokens, boundary), expected, atol=1e-14, rtol=1e-14)
