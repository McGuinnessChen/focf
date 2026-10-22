#!/usr/bin/env python3
"""Full-data protocol adapter; original learning rules stay untouched on disk."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from methods.component_scaling import trainer as common
from methods.component_scaling_fast.learner import CachedScalingLearner
from methods.ebd_vit import trainer as ebd
from methods.cifar10_full.data import build_loaders


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--smoke-batches", type=int, default=0)
    selector.add_argument("--verify-only", action="store_true")
    extra, remaining = selector.parse_known_args(argv)
    method = remaining[remaining.index("--method") + 1]
    base = ebd if method == "block_ebd" else common
    args = base.parser().parse_args(remaining)
    if args.dataset != "cifar10" or args.validation_samples or args.train_subset_samples:
        raise ValueError("This entrypoint only accepts the full official CIFAR-10 protocol")
    if extra.smoke_batches < 0:
        raise ValueError("Negative smoke batch count")
    if not extra.smoke_batches:
        if args.epochs != 100 or (args.stop_after_epoch not in (0, 100)) or not args.evaluate_test_final:
            raise ValueError("Formal runs require 100 epochs and final official test evaluation")
    elif args.evaluate_test_final or args.resume:
        raise ValueError("Smoke runs cannot access the test split or resume formal checkpoints")

    original_hashes = base.source_hashes
    original_evaluate = base.evaluate
    original_split = base.record_split_provenance
    original_logger = base.ExperimentLogger
    original_learner = base.EBDLearner if method == "block_ebd" else base.ScalingLearner
    original_loaders = base.build_loaders
    execution_files = list((ROOT / "src/methods/cifar10_full").glob("*.py"))
    execution_files += list((ROOT / "src/methods/component_scaling_fast").glob("*.py"))

    def source_hashes():
        return {**original_hashes(), **{
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in execution_files}}

    if extra.verify_only:
        loaders = build_loaders(args, smoke_batches=extra.smoke_batches)
        data = loaders[-1]
        data.update(original_split(loaders[0], loaders[2]))
        print(json.dumps({"verified": args.run_tag, "data": data, "source_sha256": source_hashes()}))
        return 0

    started = time.perf_counter()
    progress_path = None
    learner_parent = ebd.EBDLearner if method == "block_ebd" else CachedScalingLearner

    class ProgressLearner(learner_parent):
        def __init__(self, model, config):
            if method == "block_ebd":
                super().__init__(model, config)
            else:
                super().__init__(model, config, cls_only=True)

        def step(self, images, labels):
            result = super().step(images, labels)
            if progress_path is not None and (self.steps == 1 or self.steps % 16 == 0):
                atomic_json(progress_path, {
                    "pid": os.getpid(), "steps": self.steps, "batch_loss": result[1],
                    "seconds_since_start": time.perf_counter() - started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "diagnostics": result[2], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                })
            return result

    class LoggedRun(original_logger):
        def __init__(self, *positional, **kwargs):
            nonlocal progress_path
            values = list(positional)
            values[5] = {**values[5],
                "evaluation_protocol": "50,000 official train; no holdout; 10,000 official test at epoch 100",
                "checkpoint_selection": "fixed final epoch",
                "execution_optimization": "cached_cls" if method == "component" else "preserved baseline",
                "smoke_batches": extra.smoke_batches,
                "concurrency_policy": f"up to {args.jobs_per_gpu} jobs per GPU; shared timing, not efficiency evidence"}
            super().__init__(*values, **kwargs)
            progress_path = self.run_dir / "progress.json"

    def evaluate(model, loader, *rest):
        if len(loader.dataset) == 0:
            return None, None
        return original_evaluate(model, loader, *rest)

    def split_provenance(train, validation):
        return {**original_split(train, validation),
                "split_strategy": "all official training images; empty validation split"}

    base.source_hashes = source_hashes
    base.evaluate = evaluate
    base.record_split_provenance = split_provenance
    base.ExperimentLogger = LoggedRun
    base.build_loaders = lambda parsed: build_loaders(parsed, smoke_batches=extra.smoke_batches)
    if method == "block_ebd":
        base.EBDLearner = ProgressLearner
    else:
        base.ScalingLearner = ProgressLearner
    try:
        return base.main(remaining)
    finally:
        base.source_hashes = original_hashes
        base.evaluate = original_evaluate
        base.record_split_provenance = original_split
        base.ExperimentLogger = original_logger
        base.build_loaders = original_loaders
        if method == "block_ebd":
            base.EBDLearner = original_learner
        else:
            base.ScalingLearner = original_learner


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
