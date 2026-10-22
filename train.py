#!/usr/bin/env python3
"""Train one of the configurations used in the paper."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Two small training batches, without test-set evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved arguments")
    parser.add_argument("--resume", help="Resume a checkpoint produced by this release")
    options = parser.parse_args()
    config = json.loads(Path(options.config).read_text())
    args = dict(config["args"])
    if options.device:
        args["device"] = options.device
    if options.resume:
        args["resume"] = str(Path(options.resume).resolve())
    os.chdir(ROOT)
    diagnostic = config.get("protocol") == "diagnostic_subset"
    full_cifar10 = args["dataset"] == "cifar10" and not diagnostic
    method = config["method"]
    if diagnostic:
        module = config["entrypoint"]
    elif full_cifar10:
        module = "methods.cifar10_random_dictionary.trainer" if method == "random_dictionary" else "methods.cifar10_full.trainer"
    else:
        module = "methods.ebd_vit.trainer" if method == "ebd" else "methods.component_scaling.trainer"
    trainer = importlib.import_module(module)
    if full_cifar10:
        base = trainer.full if method == "random_dictionary" else trainer
        parser_module = base.ebd if method == "ebd" else base.common
    else:
        parser_module = trainer
    if options.smoke:
        if options.resume:
            parser.error("--smoke and --resume cannot be combined")
        args.update(epochs=1, stop_after_epoch=1, batch_size=2, num_workers=0,
                    embed_dim=8, num_heads=2, ffn_dim=12, evaluate_test_final=False,
                    run_tag=config["id"] + "-smoke", checkpoint_every=1)
        if not full_cifar10:
            args.update(train_subset_samples=4, validation_samples=4)
        for key, value in {"mirror_warmup":1, "mirror_probe_count":2,
                           "validation_probes":2, "output_probes":2,
                           "calibration_batch_size":2}.items():
            if key in args:
                args[key] = value
    argv = []
    for key, value in args.items():
        if value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)])
    parser_module.parser().parse_args(argv)
    if full_cifar10 and options.smoke:
        argv += ["--smoke-batches", "2"]
    if options.dry_run:
        print(json.dumps({"entrypoint": module, "argv": argv}, indent=2))
        return 0

    # The cached suffix changes execution only; local updates and random streams
    # use the paper implementations, checked against their uncached counterparts.
    if not diagnostic and method == "focf" and not full_cifar10:
        from methods.component_scaling_fast.learner import CachedScalingLearner
        class PaperLearner(CachedScalingLearner):
            def __init__(self, model, config):
                super().__init__(model, config, cls_only=True)
        trainer.ScalingLearner = PaperLearner
    if method == "random_dictionary":
        from methods.random_dictionary_fast.learner import FastRandomDictionaryLearner
        trainer.RandomDictionaryLearner = FastRandomDictionaryLearner

    original_hashes = parser_module.source_hashes
    original_logger = parser_module.ExperimentLogger
    run_dirs = []
    class PublishedRun(original_logger):
        def __init__(self, *positional, **keywords):
            values = list(positional)
            if not diagnostic and method in ("focf", "random_dictionary"):
                values[5] = {**values[5], "execution_optimization": "cached_cls"}
            if method == "random_dictionary":
                values[5]["fidelity"] = "Original random dictionary, probe stream and local updates; cached CLS suffix evaluation"
            super().__init__(*values, **keywords)
            run_dirs.append(self.run_dir)
    def release_hashes():
        paths = [Path(__file__)]
        paths += list((ROOT / "src/methods/component_scaling_fast").glob("*.py"))
        paths += list((ROOT / "src/methods/random_dictionary_fast").glob("*.py"))
        return {**original_hashes(), **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    parser_module.source_hashes = release_hashes
    parser_module.ExperimentLogger = PublishedRun
    try:
        result = trainer.main(argv)
        if not options.smoke and run_dirs:
            checkpoint = run_dirs[-1] / "artifacts/last_checkpoint.pt"
            if checkpoint.is_file():
                link = ROOT / "checkpoints" / (config["id"] + ".pt")
                link.parent.mkdir(exist_ok=True)
                if link.is_symlink():
                    link.unlink()
                if not link.exists():
                    link.symlink_to(os.path.relpath(checkpoint, link.parent))
        return result
    finally:
        parser_module.source_hashes = original_hashes
        parser_module.ExperimentLogger = original_logger


if __name__ == "__main__":
    raise SystemExit(main())
