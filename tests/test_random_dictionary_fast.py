import copy
import unittest
from unittest.mock import patch

import torch

from methods.cifar10_random_dictionary.trainer import RandomDictionaryLearner
from methods.random_dictionary_fast.learner import FastRandomDictionaryLearner
from methods.component_scaling.learner import ScalingConfig
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.configs import ViTConfig


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


class RandomDictionaryFastTest(unittest.TestCase):
    def test_cleanup_preserves_original_updates_and_checkpoint(self):
        torch.set_num_threads(2)
        for depth, codes in ((1, 0), (3, 0), (4, 2)):
            with self.subTest(depth=depth, codes=codes):
                torch.manual_seed(713)
                model = AuditedViT(ViTConfig(image_size=8, patch_size=4,
                    embed_dim=8, num_heads=2, depth=depth, ffn_dim=12, num_classes=5), 'post').double()
                cfg = ScalingConfig(method='component', output_probes=codes,
                    mirror_warmup=2, mirror_probe_count=4, validation_probes=2)
                ref = RandomDictionaryLearner(copy.deepcopy(model), cfg)
                fast = FastRandomDictionaryLearner(copy.deepcopy(model), cfg)
                error = AssertionError('No reverse autograd in random dictionary')
                with patch.object(torch.autograd, 'grad', side_effect=error), \
                        patch.object(torch.Tensor, 'backward', side_effect=error):
                    for size in (3, 1):
                        x = torch.randn(size, 3, 8, 8, dtype=torch.float64)
                        y = torch.arange(size) % 5
                        equal(ref.step(x, y), fast.step(x, y))
                        equal(ref.state_dict(), fast.state_dict())
                    restored = FastRandomDictionaryLearner(copy.deepcopy(model), cfg)
                    restored.load_state_dict(copy.deepcopy(ref.state_dict()))
                    equal(ref.step(x, y), restored.step(x, y))
                    equal(ref.state_dict(), restored.state_dict())


if __name__ == '__main__':
    unittest.main()
