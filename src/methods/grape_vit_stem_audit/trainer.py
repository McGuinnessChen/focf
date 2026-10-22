#!/usr/bin/env python3
"""Isolated GrAPE ViT normalization/scope audit; not the authors' code."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from experiment_logging import ExperimentLogger
from reproducibility import configure_reproducibility
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.trainer import build_loaders, str2bool
from methods.grape_vit_audit.model import AuditedViT
from methods.grape_vit_stem_audit.learner import GroupedStemGrAPELearner
from methods.grape.core import calibration_due
from methods.grape.learner import GrAPEConfig, MacroGrAPELearner, objective


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet"], default="cifar10")
    p.add_argument("--data-root")
    p.add_argument("--norm-layout", choices=["pre", "post"], default="post")
    p.add_argument("--method", choices=["grape", "dfa", "bp"], default="grape")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feedback-seed", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--stop-after-epoch", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-subset-samples", type=int, default=5000)
    p.add_argument("--validation-samples", type=int, default=1000)
    p.add_argument("--validation-split-seed", type=int, default=2027)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--augmentation", type=str2bool, default=True)
    p.add_argument("--evaluate-test-final", type=str2bool, default=False)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--loss", choices=["cross_entropy", "mse"], default="cross_entropy")
    p.add_argument("--parameter-scope", choices=["blocks_head", "all"], default="blocks_head")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--feedback-column-lr", type=float, default=0.01)
    p.add_argument("--aggregation", choices=["mean_estimate", "mean_sample_cosine"], default="mean_estimate")
    p.add_argument("--calibration-interval", type=int, default=0)
    p.add_argument("--calibration-lr", type=float, default=0.01)
    p.add_argument("--calibration-momentum", type=float, default=0.9)
    p.add_argument("--calibration-batch-size", type=int, default=64)
    p.add_argument("--run-tag", default="grape_development")
    p.add_argument("--require-clean-git", type=str2bool, default=False)
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume")
    return p


def move(batch, device, dtype):
    x, y = batch
    return x.to(device=device, dtype=dtype, non_blocking=True), y.to(device=device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, device, dtype, loss_name):
    model.eval()
    total = correct = 0
    loss_sum = 0.
    for batch in loader:
        x, y = move(batch, device, dtype)
        logits = model(x)
        loss = objective(logits, y, loss_name)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        total += len(y)
        correct += int((logits.argmax(1) == y).sum())
        loss_sum += float(loss) * len(y)
    return correct / total, loss_sum / total


def capture_rng(loader, calibration_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state(), "calibration": calibration_generator.get_state()}


def restore_rng(state, loader, calibration_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())
    calibration_generator.set_state(state["calibration"].cpu())


def calibration_batch(dataset, count, generator):
    # Select and augment only training examples. Isolate every augmentation RNG
    # so adding a calibration event cannot change the normal training stream.
    indices = torch.randperm(len(dataset), generator=generator)[:min(count, len(dataset))].tolist()
    seed = int(torch.randint(0, 2**31, (), generator=generator))
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            return default_collate([dataset[i] for i in indices])
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def protocol(args):
    # Runtime/output choices may differ on resume; scientific settings may not.
    ignored = {"stop_after_epoch", "resume", "run_tag", "require_clean_git", "jobs_per_gpu", "checkpoint_every"}
    return {k: v for k, v in vars(args).items() if k not in ignored}


def source_hashes():
    paths = list((PROJECT_ROOT / "src/methods/grape").glob("*.py"))
    paths += [PROJECT_ROOT / "src/methods/transformer_ebd" / name for name in
              ("vit_model.py", "configs.py", "trainer.py", "class_addressed_topology.py")]
    paths += list((PROJECT_ROOT / "src/methods/grape_vit_audit").glob("*.py"))
    paths += list((PROJECT_ROOT / "src/methods/grape_vit_stem_audit").glob("*.py"))
    paths += [PROJECT_ROOT / "src/reproducibility.py", PROJECT_ROOT / "src/experiment_logging.py"]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs < 1 or args.checkpoint_every < 1 or args.calibration_interval < 0:
        raise ValueError("Invalid epoch/checkpoint/calibration setting")
    if args.method == "bp" and args.calibration_interval:
        raise ValueError("No extra calibration for full BP reference")
    stop_epoch = args.stop_after_epoch or args.epochs
    if not 1 <= stop_epoch <= args.epochs:
        raise ValueError("stop-after-epoch must be inside the scheduled run")
    if args.evaluate_test_final and stop_epoch != args.epochs:
        raise ValueError("Development screens cannot request final test evaluation")
    torch.set_num_threads(args.threads)
    reproducibility = configure_reproducibility(args.seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    train, train_eval, validation, test, data_info = build_loaders(args)
    cfg = ViTConfig(image_size=args.image_size, patch_size=args.patch_size,
                    num_classes=data_info["num_classes"], embed_dim=args.embed_dim,
                    depth=args.depth, num_heads=args.num_heads, ffn_dim=args.ffn_dim)
    model = AuditedViT(cfg, norm_layout=args.norm_layout).to(device=device, dtype=dtype)
    learning_cfg = GrAPEConfig(**{k: getattr(args, k) for k in GrAPEConfig.__dataclass_fields__})
    learner = GroupedStemGrAPELearner(model, learning_cfg)
    calibration_generator = torch.Generator().manual_seed(args.seed + 71831)
    hashes = source_hashes()
    start_epoch, previous_records, elapsed_before = 1, [], 0.
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != protocol(args) or checkpoint["data_info"] != data_info:
            raise ValueError("Resume protocol or data split differs")
        if checkpoint["source_sha256"] != hashes:
            raise ValueError("Resume source files differ; preserve the original implementation")
        learner.load_state_dict(checkpoint["learner"])
        start_epoch = checkpoint["epoch"] + 1
        previous_records = checkpoint["records"]
        elapsed_before = checkpoint["training_wall_clock_sec"]
        if start_epoch > stop_epoch:
            raise ValueError("Checkpoint already reached requested stop epoch")
    run_config = {"args": vars(args), "model": cfg.to_dict(), "learning": asdict(learning_cfg),
                  "data": data_info, "reproducibility": reproducibility, "source_sha256": hashes,
                  "feedback_eta_B": args.feedback_column_lr * cfg.token_count * cfg.embed_dim,
                  "classification_head_trainable": True, "optimizer_schedule": "constant",
                  "calibration_optimizer_state": "separate SGD momentum",
                  "fidelity": "independent GrAPE ViT audit; embedding in first macro block; not author code",
                  "norm_layout": args.norm_layout, "stem_grouping": "first_encoder_block",
                  "normalization_control": "same initial weights and final readout LN; only within-block LN location changes",
                  "concurrency_policy": "development run; one assigned job per GPU; no paper timing claim",
                  "parent_checkpoint": args.resume}
    logger = ExperimentLogger(PROJECT_ROOT, args.dataset, f"vit_{args.norm_layout}ln_d{cfg.depth}_p{cfg.patch_size}_e{cfg.embed_dim}",
                              "grape_stem_audit_" + args.method, args.seed, run_config,
                              run_tag=args.run_tag, require_clean_git=args.require_clean_git)
    logger.reset_peak_cuda_memory()
    if checkpoint is not None:
        restore_rng(checkpoint["rng"], train, calibration_generator)
    records = list(previous_records)
    start = time.perf_counter()
    for epoch in range(start_epoch, stop_epoch + 1):
        epoch_start = time.perf_counter()
        count = correct = 0
        loss_sum = 0.
        sums = defaultdict(float)
        try:
            for batch in train:
                x, y = move(batch, device, dtype)
                logits, loss, diagnostics = learner.step(x, y)
                count += len(y)
                correct += int((logits.argmax(1) == y).sum())
                loss_sum += loss * len(y)
                for key, value in diagnostics.items():
                    sums[key] += value * len(y)
            calibration_loss = None
            if calibration_due(epoch, args.calibration_interval):
                batch = calibration_batch(train.dataset, args.calibration_batch_size, calibration_generator)
                calibration_loss = learner.calibrate(*move(batch, device, dtype))
            val_accuracy, val_loss = evaluate(model, validation, device, dtype, args.loss)
            diagnostics = {key: value / count for key, value in sums.items()}
            record = logger.log_epoch(
                epoch=epoch, train_loss=loss_sum/count, train_accuracy=correct/count,
                test_loss=None, test_accuracy=None, epoch_time_sec=time.perf_counter()-epoch_start,
                learning_rate=learner.optimizer.param_groups[0]["lr"],
                extra={"validation_accuracy": val_accuracy, "validation_loss": val_loss,
                       "test_accessed": False, "calibration_loss": calibration_loss,
                       "bp_calibration_batches": learner.calibration_batches,
                       "training_steps": learner.steps, "diagnostics": diagnostics},
            )
            records.append(record)
            print(json.dumps({k: record[k] for k in ("epoch", "train_loss", "train_accuracy",
                                                    "validation_loss", "validation_accuracy", "epoch_time_sec")}), flush=True)
            if epoch % args.checkpoint_every == 0 or epoch == stop_epoch:
                atomic_checkpoint(logger.artifacts_dir / "last_checkpoint.pt", {
                    "epoch": epoch, "protocol": protocol(args), "data_info": data_info,
                    "source_sha256": hashes, "learner": learner.state_dict(), "records": records,
                    "rng": capture_rng(train, calibration_generator),
                    "training_wall_clock_sec": elapsed_before + time.perf_counter() - start,
                })
        except Exception as error:
            logger.log_epoch(epoch=epoch, train_loss=None, train_accuracy=None, test_loss=None, test_accuracy=None,
                             epoch_time_sec=time.perf_counter()-epoch_start, learning_rate=args.learning_rate,
                             failed=True, failure_reason=repr(error), extra={"test_accessed": False})
            raise
    train_accuracy, train_loss = evaluate(model, train_eval, device, dtype, args.loss)
    test_accuracy = test_loss = None
    if test is not None:
        test_accuracy, test_loss = evaluate(model, test, device, dtype, args.loss)
    summary = {"completed_epoch": stop_epoch, "scheduled_epochs": args.epochs,
               "status": "completed" if stop_epoch == args.epochs else "development_stage_completed",
               "seed": args.seed, "method": args.method, "train_accuracy": train_accuracy,
               "train_loss": train_loss, "validation_accuracy": records[-1]["validation_accuracy"],
               "validation_loss": records[-1]["validation_loss"], "test_accuracy": test_accuracy,
               "test_loss": test_loss, "test_accessed": test is not None,
               "bp_calibration_batches": learner.calibration_batches,
               "wall_clock_time_sec": elapsed_before + time.perf_counter() - start,
               "peak_gpu_memory_mb": logger.peak_cuda_memory_mb(),
               "last_checkpoint": str(logger.artifacts_dir / "last_checkpoint.pt"),
               "classification_head_trainable": True, "parameter_scope": args.parameter_scope,
               "norm_layout": args.norm_layout, "stem_grouping": "first_encoder_block",
               "comparison_warning": "Different head, loss and optimizer from existing FOCF screen; no direct superiority inference"}
    (logger.artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
