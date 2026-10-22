"""Drop-in learner with the identical scientific config and checkpoint schema."""
from methods.component_scaling.learner import ScalingLearner
from .feedback import CachedComponentFeedback


class CachedScalingLearner(ScalingLearner):
    def __init__(self, model, config, *, cls_only=False):
        super().__init__(model, config)
        if config.method == "component":
            self.component = CachedComponentFeedback(model, config, cls_only=cls_only)
