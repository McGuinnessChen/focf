"""Independent checks of sparse BP anchors in the preserved ViT adaptation."""
import copy
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from methods.grape.core import calibration_due
from methods.grape.learner import GrAPEConfig
from methods.grape_vit_audit.model import AuditedViT
from methods.grape_vit_stem_audit.learner import GroupedStemGrAPELearner
from methods.grape_vit_stem_audit.trainer import calibration_batch
from methods.transformer_ebd.configs import ViTConfig


def learner():
    torch.manual_seed(20261012)
    model = AuditedViT(ViTConfig(image_size=8, patch_size=4, embed_dim=8,
                               num_heads=2, ffn_dim=12, depth=2,
                               num_classes=3), "post").double()
    return GroupedStemGrAPELearner(model, GrAPEConfig(
        parameter_scope="all", feedback_column_lr=0.1,
        calibration_lr=0.001, calibration_momentum=0.9))


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("batch_size", [3, 128])
def test_calibration_matches_independent_full_bp_sgd_and_preserves_other_state(batch_size):
    actual = learner()
    train_x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    train_y = torch.tensor([0, 1, 2])
    for event in range(2):
        # Leave nonempty Adam state, and preserve SGD momentum between anchors.
        actual.step(train_x, train_y)
        before = copy.deepcopy(actual.state_dict())
        oracle = copy.deepcopy(actual.model)
        params = list(oracle.parameters())
        assert all(p.requires_grad for p in params)
        reference = torch.optim.SGD(params, lr=0.001, momentum=0.9,
                                    weight_decay=1e-4)
        reference.load_state_dict(copy.deepcopy(before["calibration_optimizer"]))
        x = torch.randn(batch_size, 3, 8, 8, dtype=torch.float64)
        y = torch.arange(batch_size) % 3
        reference.zero_grad(set_to_none=True)
        expected_loss = F.cross_entropy(oracle(x), y, reduction="mean")
        expected_loss.backward()
        expected_gradients = {n: p.grad.clone() for n, p in oracle.named_parameters()}
        reference.step()
        # Calibration must clear stale local gradients before full BP.
        for p in actual.model.parameters():
            p.grad = torch.full_like(p, 123.0)
        loss = actual.calibrate(x, y)
        assert loss == float(expected_loss.detach())
        for name, p in actual.model.named_parameters():
            torch.testing.assert_close(p.grad, expected_gradients[name], rtol=0, atol=0)
            assert torch.count_nonzero(p.grad) > 0, name
        same_tree(actual.model.state_dict(), oracle.state_dict())
        same_tree(actual.calibration_optimizer.state_dict(), reference.state_dict())
        after = actual.state_dict()
        for key in ["optimizer", "feedback", "initial_feedback", "probe_rng", "steps"]:
            same_tree(before[key], after[key])
        assert actual.calibration_batches == event + 1


def test_calibration_mean_loss_does_not_scale_with_duplicated_batch():
    a = learner()
    b = learner()
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    y = torch.tensor([0, 1, 2])
    a.calibrate(x, y)
    b.calibrate(x.repeat(4, 1, 1, 1), y.repeat(4))
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-13)


def test_checkpoint_restores_calibration_momentum_and_next_mixed_updates():
    a = learner()
    x = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    y = torch.tensor([0, 1, 2])
    a.step(x, y)
    a.calibrate(x, y)
    state = copy.deepcopy(a.state_dict())
    b = learner()
    b.load_state_dict(state)
    a.step(x, y)
    b.step(x, y)
    a.calibrate(x, y)
    b.calibrate(x, y)
    same_tree(a.state_dict(), b.state_dict())


class RandomizedTrainingDataset:
    def __len__(self):
        return 256

    def __getitem__(self, index):
        return torch.tensor([random.random(), np.random.random(),
                             float(torch.rand(()))]), index


def test_calibration_samples_unique_training_items_and_isolates_augmentation_rng():
    random.seed(37)
    np.random.seed(38)
    torch.manual_seed(39)
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(71831)
    saved = generator.get_state().clone()
    x, ids = calibration_batch(RandomizedTrainingDataset(), 128, generator)
    assert len(ids) == len(ids.unique()) == 128
    assert int(ids.min()) >= 0 and int(ids.max()) < 256
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert not torch.equal(generator.get_state(), saved)
    generator.set_state(saved)
    other_x, other_ids = calibration_batch(RandomizedTrainingDataset(), 128, generator)
    same_tree((x, ids), (other_x, other_ids))


def test_epoch_end_calibration_counts_for_actual_baseline_schedules():
    assert [e for e in range(1, 101) if calibration_due(e, 10)] == list(range(10, 101, 10))
    assert sum(calibration_due(e, 1) for e in range(1, 101)) == 100
    assert not any(calibration_due(e, 0) for e in range(1, 101))
