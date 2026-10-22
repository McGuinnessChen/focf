#!/usr/bin/env python3
"""Reuse the original random dictionary with the formal CIFAR-10 training loop.

Only execution/provenance and checkpoint restoration are added here. The random
dictionary, its RNG stream, response queries, fitting, and local updates are
imported unchanged from the earlier control.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_random_route_control import RandomRouteFeedback
from methods.cifar10_full import trainer as full
from methods.component_postln.learner import CommonLearner

CONTROL = "random_route_equal_mode_control"


class ResumableRandomRouteFeedback(RandomRouteFeedback):
    def state_dict(self):
        return {**super().state_dict(), "feedback_kind": CONTROL}

    def load_state_dict(self, state):
        if state.get("feedback_kind") != CONTROL:
            raise ValueError("Expected a random-dictionary checkpoint")
        self.mirrors.load_state_dict(state["mirrors"])
        self.generator.set_state(state["operator_rng"].cpu())


class RandomDictionaryLearner(CommonLearner):
    def __init__(self, model, config, *, cls_only=False):
        # cls_only is accepted for the full-data progress adapter's interface.
        # The preserved random control evaluates the complete suffix unchanged.
        if config.method != "component":
            raise ValueError("The random control requires component local updates")
        super().__init__(model, config)
        self.component = ResumableRandomRouteFeedback(model, config)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in argv or argv[argv.index("--method") + 1] != "component":
        raise ValueError("Use --method component for the shared local training loop")
    original_parent = full.CachedScalingLearner
    original_hashes = full.common.source_hashes
    original_protocol = full.common.protocol
    original_logger = full.common.ExperimentLogger
    run_dirs = []

    def source_hashes():
        paths = [ROOT / "scripts/train_random_route_control.py"]
        paths += list(Path(__file__).parent.glob("*.py"))
        return {**original_hashes(), **{
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}}

    def protocol(args):
        return {**original_protocol(args), "feedback_control": CONTROL}

    class RandomRun(original_logger):
        def __init__(self, *positional, **kwargs):
            values = list(positional)
            values[3] = "component_random_dictionary"
            config = dict(values[5])
            config.update({
                "method": CONTROL,
                "control": "Per-sample random dictionary; same carrier, mode count, norm, output codes, ridge, trust and acceptance rule as FOCF",
                "fidelity": "Original train_random_route_control.RandomRouteFeedback reused without changes",
                "execution_optimization": "original full-minibatch random control; cached_cls and execution chunk flags are unused",
                "labels_used_for_route_fit": False,
                "bp_calibration_batches": 0,
                "feedback_rng": "Original shared RNG for random dictionary, output codes and acceptance probes; saved in checkpoint",
                "component_feedback": {
                    **config["component_feedback"],
                    "dictionary": "Resampled per image and step; each mode normalized to the carrier Frobenius norm",
                    "cross_block_expansion": "Random modes replacing the entire structured dictionary; common readout carrier retained",
                    "channel_maps": "Original mirror updates retained; their maps do not set random dictionary values",
                    "compute_budget_matched": True,
                    "feedback_sample_chunk": "full minibatch (original random control)",
                    "output_query_chunk": "all output codes (original random control)",
                },
            })
            values[5] = config
            super().__init__(*values, **kwargs)
            run_dirs.append(self.run_dir)

    full.CachedScalingLearner = RandomDictionaryLearner
    full.common.source_hashes = source_hashes
    full.common.protocol = protocol
    full.common.ExperimentLogger = RandomRun
    try:
        result = full.main(argv)
        for run_dir in run_dirs:
            path = run_dir / "artifacts/summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                summary.update(method=CONTROL, comparison_warning=
                               "Random dictionary control with matched logical query budget; runtime includes its original implementation")
                full.atomic_json(path, summary)
        return result
    finally:
        full.CachedScalingLearner = original_parent
        full.common.source_hashes = original_hashes
        full.common.protocol = original_protocol
        full.common.ExperimentLogger = original_logger


if __name__ == "__main__":
    raise SystemExit(main())
