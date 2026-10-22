"""Use the original local gradients, optimizer and checkpoint format."""
from methods.component_postln.learner import CommonLearner
from .feedback import FastRandomRouteFeedback


class FastRandomDictionaryLearner(CommonLearner):
    def __init__(self, model, config, *, cls_only=False):
        if config.method != "component":
            raise ValueError("Random dictionary requires component local updates")
        super().__init__(model, config)
        self.component = FastRandomRouteFeedback(model, config, cls_only=cls_only)
