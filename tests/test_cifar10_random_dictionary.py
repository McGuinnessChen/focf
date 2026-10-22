"""Verify preservation of the earlier control and exact continuation on resume."""
import copy
import unittest
from unittest.mock import patch

import torch

from methods.cifar10_random_dictionary.trainer import (
    CONTROL, RandomDictionaryLearner, RandomRouteFeedback,
)
from methods.component_postln.learner import CommonLearner
from methods.component_scaling.learner import ScalingConfig
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.configs import ViTConfig


def model():
    torch.manual_seed(12012)
    return AuditedViT(ViTConfig(image_size=8, patch_size=4, embed_dim=8,
        num_heads=2, depth=3, ffn_dim=12, num_classes=5), "post").double()


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def check_original_updates_and_resumed_rng_are_preserved():
    torch.set_num_threads(2)
    cfg = ScalingConfig(method="component", output_probes=0, mirror_warmup=2,
                        mirror_probe_count=4, validation_probes=2)
    reference = CommonLearner(model(), cfg)
    reference.component = RandomRouteFeedback(reference.model, cfg)
    actual = RandomDictionaryLearner(model(), cfg)
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    y = torch.tensor([0, 1, 2])

    for _ in range(2):
        a, b = actual.step(x, y), reference.step(x, y)
        assert_equal(a, b)
        state = copy.deepcopy(actual.state_dict())
        assert state["component"].pop("feedback_kind") == CONTROL
        assert_equal(state, reference.state_dict())
        for p, q in zip(actual.model.parameters(), reference.model.parameters()):
            assert_equal(p.grad, q.grad)
        assert a[2]["boundary0_operator_modes"] == 17
        assert a[2]["boundary1_operator_modes"] == 9
        assert a[2]["suffix_directional_evaluations"] == 2 * ((17 + 9) * 5 + 2 * 2)
        assert actual.calibration_batches == 0

    restored = RandomDictionaryLearner(model(), cfg)
    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
    assert_equal(restored.step(x, y), actual.step(x, y))
    assert_equal(restored.state_dict(), actual.state_dict())

    invalid = copy.deepcopy(actual.state_dict())
    invalid["component"]["feedback_kind"] = "structured_dictionary"
    with unittest.TestCase().assertRaisesRegex(ValueError, "random-dictionary checkpoint"):
        restored.load_state_dict(invalid)


class RandomDictionaryTest(unittest.TestCase):
    def test_original_updates_and_resumed_rng_are_preserved(self):
        error = AssertionError("Random control must not introduce reverse autograd")
        with patch.object(torch.autograd, "grad", side_effect=error), \
                patch.object(torch.Tensor, "backward", side_effect=error):
            check_original_updates_and_resumed_rng_are_preserved()


if __name__ == "__main__":
    unittest.main()
