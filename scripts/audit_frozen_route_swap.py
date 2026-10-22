#!/usr/bin/env python3
"""Swap fitted corrections between inputs without refitting or reacceptance.

Reconstruct the training trace-fit rule with a saved, frozen mirror bank.
This is a checkpoint diagnostic, not a replay of an original training batch.
Keep recipient M_i and ||Delta_i||_F, replace only the direction of Delta_i
by that of another active sample, and test on independent forward probes.
No labels, BP targets, or parameter updates are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methods.component_postln.feedback import ForwardMirrorBank
from methods.component_postln.learner import CommonConfig
from methods.component_postln.operators import downstream_component_dictionary
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.component_route_operator import (
    batched_vector_responses, operator_response, sample_output_codes,
    select_operator, trace_component_operator,
)
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.forward_response_credit import (
    forward_vector_responses, sample_probes,
)
from methods.transformer_ebd.trainer import build_loaders


def tensor_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


@torch.no_grad()
def swap_correction(correction):
    """Derange nonzero corrections; zero corrections stay zero in both arms."""
    norms = correction.flatten(1).norm(dim=-1)
    active = torch.nonzero(norms > 1e-20).flatten()
    eligible = torch.zeros_like(norms, dtype=torch.bool)
    swapped = correction.clone()
    if len(active) >= 2:
        donor = active.roll(1)
        swapped[active] = correction[donor] * (
            norms[active] / norms[donor]
        )[:, None, None, None]
        eligible[active] = True
    error = (swapped.flatten(1).norm(dim=-1) - norms).abs() / norms.clamp_min(1e-20)
    if not torch.isfinite(swapped).all() or float(error.max()) > 1e-5:
        raise RuntimeError("The swap failed the correction-norm invariant")
    return swapped, eligible, float(error.max())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    cfg = json.loads(path.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    checkpoint = Path(cfg["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu")
    model_cfg = ViTConfig(image_size=32, patch_size=4, num_classes=10,
                          embed_dim=128, depth=4, num_heads=4, ffn_dim=256)
    model = AuditedViT(model_cfg, norm_layout="post").to(device).float()
    model.load_state_dict(state["learner"]["model"], strict=True)
    model.eval()
    learning = CommonConfig(**state["learner"]["config"])
    mirrors = ForwardMirrorBank(model, learning)
    mirrors.load_state_dict(state["learner"]["component"]["mirrors"])
    before = tensor_digest({"model": model.state_dict(), "mirrors": mirrors.state_dict()})
    loader_args = type("Args", (), {
        "dataset": "cifar10", "data_root": "cifar10_data", "augmentation": False,
        "validation_samples": 1000, "train_subset_samples": 5000,
        "validation_split_seed": 2027, "seed": cfg["seed"],
        "batch_size": cfg["batch_size"], "num_workers": 2,
        "device": "cuda", "evaluate_test_final": False,
    })()
    loaders = build_loaders(loader_args)
    validation, data_info = loaders[2], loaders[-1]
    fit_rng = torch.Generator(device=device).manual_seed(99117 + cfg["seed"])
    heldout_rng = torch.Generator(device=device).manual_seed(199117 + cfg["seed"])
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for batch_index, (images, _) in enumerate(validation):
        if batch_index >= cfg["eval_batches"]:
            break
        caches = model.forward_with_cache(images.to(device)).block_caches
        for boundary in range(model_cfg.depth - 1):
            hidden = caches[boundary].block_output

            def suffix(tokens, boundary=boundary):
                for block in model.blocks[boundary + 1:]:
                    tokens, _ = block(tokens)
                return model.head(model.final_norm(tokens)[:, 0])

            carrier, dictionary, names = downstream_component_dictionary(
                model, caches, mirrors.maps, boundary)
            codes = sample_output_codes(
                carrier, learning.output_probes or model_cfg.num_classes, fit_rng)
            candidate, _, _ = trace_component_operator(
                suffix, hidden, carrier, dictionary, codes,
                epsilon=learning.probe_epsilon, ridge=learning.operator_ridge,
                trust_ratio=learning.trust_ratio, batch_output_queries=True)
            accept_probes = sample_probes(
                hidden, learning.validation_probes, fit_rng, orthogonal=True)
            accept_response = forward_vector_responses(
                suffix, hidden, accept_probes, learning.probe_epsilon)
            matched, accepted = select_operator(
                carrier, candidate, accept_probes, accept_response)
            correction = matched - carrier
            exchanged, eligible, norm_error = swap_correction(correction)
            # Deliberately do not fit or accept the donor correction on recipient data.
            swapped = carrier + exchanged
            heldout = sample_probes(
                hidden, cfg["heldout_probes"], heldout_rng, orthogonal=True)
            response = batched_vector_responses(
                suffix, hidden, heldout, learning.probe_epsilon)
            errors = {
                name: (operator_response(operator, heldout) - response).square().mean((1, 2))
                for name, operator in (("identity", carrier), ("matched", matched),
                                       ("swapped", swapped))
            }
            relative = {name: (error / errors["identity"].clamp_min(1e-20)).tolist()
                        for name, error in errors.items()}
            if not all(torch.isfinite(error).all() for error in errors.values()):
                raise RuntimeError("Nonfinite held-out error")
            rows.append({
                "batch": batch_index, "boundary": boundary, "samples": len(images),
                "modes": len(names), "accepted": accepted.tolist(),
                "swap_eligible": eligible.tolist(), "max_relative_norm_error": norm_error,
                "relative_mse_per_sample": relative,
                "raw_mse_per_sample": {name: error.tolist() for name, error in errors.items()},
            })
        print(json.dumps({"batch": batch_index + 1, "elapsed_seconds": time.perf_counter() - start}), flush=True)
    after = tensor_digest({"model": model.state_dict(), "mirrors": mirrors.state_dict()})
    if before != after:
        raise RuntimeError("Frozen model or mirror state changed")
    if not rows:
        raise RuntimeError("No samples evaluated")
    metrics = {}
    for subset in ("all", "swap_eligible"):
        metrics[subset] = {}
        for name in ("identity", "matched", "swapped"):
            values = [value for row in rows
                      for value, eligible in zip(row["relative_mse_per_sample"][name], row["swap_eligible"])
                      if subset == "all" or eligible]
            metrics[subset][name] = sum(values) / len(values) if values else None
        metrics[subset]["sample_boundary_pairs"] = sum(
            row["samples"] if subset == "all" else sum(row["swap_eligible"]) for row in rows)
    result = {
        "status": "completed", "seed": cfg["seed"], "config": cfg,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": state["epoch"], "data": data_info,
        "model": model_cfg.to_dict(), "learning": vars(learning),
        "frozen_state_before": before, "frozen_state_after": after,
        "metrics": metrics, "rows": rows,
        "wall_clock_seconds": time.perf_counter() - start,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "protocol": {
            "validation_only": True, "labels_used": False, "parameter_updates": 0,
            "mirror_updates": 0, "operator_reconstruction": "Training trace fit and acceptance with saved frozen mirrors; fresh probes, not original minibatch replay",
            "swap": "Cyclic derangement among nonzero accepted corrections within each batch and boundary",
            "recipient_carrier_preserved": True, "recipient_correction_norm_preserved": True,
            "refit_after_swap": False, "reaccept_after_swap": False,
            "heldout_probe_rng_independent": True, "aggregation": "Sample-weighted over all three boundaries",
            "accuracy_and_loss": "Not applicable: frozen response diagnostic, not classifier evaluation",
        },
        "interpretation": "Input correspondence of fitted corrections at fixed checkpoints; does not by itself establish a primary training bottleneck.",
    }
    out = ROOT / cfg["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
