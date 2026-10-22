"""Execution-only chunking around the preserved post-LN learning rule."""
from dataclasses import dataclass

from methods.component_postln.learner import CommonConfig, CommonLearner
from .feedback import ChunkedComponentFeedback


@dataclass(frozen=True)
class ScalingConfig(CommonConfig):
    output_probes: int = 16
    feedback_sample_chunk: int = 16
    output_query_chunk: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.feedback_sample_chunk < 1 or self.output_query_chunk < 1:
            raise ValueError("Chunk sizes must be positive")


class ScalingLearner(CommonLearner):
    def __init__(self, model, config):
        super().__init__(model, config)
        if config.method == "component":
            self.component = ChunkedComponentFeedback(model, config)
