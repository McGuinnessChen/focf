#!/usr/bin/env python3
"""Measure output-code budgets at frozen, formally trained FOCF checkpoints.

The saved model and mirror bank never change. All budgets share the complete
class-space dictionary/Gram, acceptance probes and independent evaluation
probes. Prefixes of one random orthogonal code set pair the sampled budgets.
The complete coordinate basis is a deterministic forward-fit reference, not
an exact-gradient oracle. No derivative or classification error enters fitting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_frozen_route_swap import tensor_digest
from methods.component_postln.feedback import ForwardMirrorBank
from methods.component_postln.operators import downstream_component_dictionary
from methods.component_scaling.feedback import chunked_vector_responses
from methods.component_scaling.learner import ScalingConfig
from methods.component_scaling_fast.feedback import cls_suffix, fit_with_gram
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.component_route_operator import (
    operator_response, sample_output_codes, select_operator,
)
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.forward_response_credit import sample_probes
from methods.transformer_ebd.trainer import build_loaders


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_rows(rows):
    """Equal image/repetition/boundary weight, including a short final batch."""
    result = {}
    for key in sorted({row["budget_label"] for row in rows}):
        selected = [row for row in rows if row["budget_label"] == key]
        count = sum(row["samples"] for row in selected)
        result[key] = {
            "relative_response_mse": sum(sum(r["relative_mse_per_image"]) for r in selected) / count,
            "acceptance_fraction": sum(sum(r["accepted"]) for r in selected) / count,
            "fit_accept_ms_per_image_all_boundaries": (
                1000 * sum(r["fit_accept_seconds"] for r in selected) / count
                * len({r["boundary"] for r in selected})),
            "image_boundary_repetition_count": count,
            "per_boundary": {},
        }
        for boundary in sorted({r["boundary"] for r in selected}):
            subset = [r for r in selected if r["boundary"] == boundary]
            n = sum(r["samples"] for r in subset)
            result[key]["per_boundary"][str(boundary)] = {
                "relative_response_mse": sum(sum(r["relative_mse_per_image"]) for r in subset) / n,
                "acceptance_fraction": sum(sum(r["accepted"]) for r in subset) / n,
            }
    if "full" in result:
        reference = result["full"]["fit_accept_ms_per_image_all_boundaries"]
        for metric in result.values():
            metric["relative_fit_accept_time"] = metric["fit_accept_ms_per_image_all_boundaries"] / reference
    return result


def timed_cuda(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    destination = ROOT / cfg["output"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    if state["epoch"] != 100:
        raise RuntimeError("The diagnostic requires the final epoch-100 checkpoint")
    protocol = state["protocol"]
    if protocol["dataset"] != cfg["dataset"] or protocol["seed"] != cfg["seed"]:
        raise RuntimeError("Checkpoint/config mismatch")
    model_cfg = ViTConfig(**{key: protocol[key] for key in (
        "image_size", "patch_size", "embed_dim", "depth", "num_heads", "ffn_dim")},
        num_classes=state["data_info"]["num_classes"])
    model = AuditedViT(model_cfg, norm_layout="post").to(device).float().eval()
    model.load_state_dict(state["learner"]["model"], strict=True)
    model.requires_grad_(False)
    learning = ScalingConfig(**state["learner"]["config"])
    mirrors = ForwardMirrorBank(model, learning)
    mirrors.load_state_dict(state["learner"]["component"]["mirrors"])
    before = tensor_digest({"model": model.state_dict(), "mirrors": mirrors.state_dict()})
    model_checkpoint_sha = sha(cfg["checkpoint"])
    loader_args = dict(protocol)
    loader_args.update(augmentation=False, num_workers=0, evaluate_test_final=False,
                       batch_size=cfg["batch_size"])
    loaders = build_loaders(SimpleNamespace(**loader_args))
    validation, data_info = loaders[2], loaders[-1]
    if data_info["validation_indices_sha256"] != state["data_info"]["validation_indices_sha256"]:
        raise RuntimeError("Diagnostic cohort does not match the saved data provenance")
    cohort = [int(i) for i in validation.dataset.indices[:cfg["samples"]]]
    budgets = cfg["budgets"]
    if not all(1 <= k < model_cfg.num_classes for k in budgets):
        raise ValueError("Sampled budgets must be smaller than the class count")
    rows, processed, correct, loss_sum = [], 0, 0, 0.0
    suffix_relative_error = 0.0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, (images, labels) in enumerate(validation):
        remaining = cfg["samples"] - processed
        if remaining <= 0:
            break
        images, labels = images[:remaining].to(device), labels[:remaining].to(device)
        batch = model.forward_with_cache(images)
        correct += int((batch.logits.argmax(-1) == labels).sum())
        loss_sum += float(torch.nn.functional.cross_entropy(batch.logits, labels, reduction="sum"))
        for boundary in range(model_cfg.depth - 1):
            hidden = batch.block_caches[boundary].block_output

            def suffix(tokens, boundary=boundary):
                return cls_suffix(model, tokens, boundary)

            suffix_logits = suffix(hidden)
            discrepancy = float((suffix_logits - batch.logits).norm() / batch.logits.norm().clamp_min(1e-20))
            suffix_relative_error = max(suffix_relative_error, discrepancy)
            if discrepancy > 3e-5:
                raise RuntimeError("Optimized suffix disagrees with the model forward pass")
            carrier, dictionary, names = downstream_component_dictionary(
                model, batch.block_caches, mirrors.maps, boundary)
            gram = dictionary.flatten(2) @ dictionary.flatten(2).transpose(1, 2)
            stream_offset = cfg["seed"] * 100000 + batch_index * 100 + boundary * 10
            heldout_rng = torch.Generator(device=device).manual_seed(cfg["heldout_seed"] + stream_offset)
            heldout = sample_probes(hidden, cfg["heldout_probes"], heldout_rng, orthogonal=True)
            heldout_response = chunked_vector_responses(
                suffix, hidden, heldout, learning.probe_epsilon, learning.output_query_chunk)
            carrier_mse = (operator_response(carrier, heldout) - heldout_response).square().mean((1, 2))
            if not bool((carrier_mse > 1e-20).all()):
                raise RuntimeError("Undefined relative response error")

            def fit(codes):
                return fit_with_gram(suffix, hidden, carrier, dictionary, codes, gram,
                    epsilon=learning.probe_epsilon, ridge=learning.operator_ridge,
                    trust_ratio=learning.trust_ratio, query_chunk=learning.output_query_chunk)

            full_codes = sample_output_codes(carrier, model_cfg.num_classes, heldout_rng)
            # Warm kernels before collecting timing, without changing saved state.
            if batch_index == 0:
                warm_rng = torch.Generator(device=device).manual_seed(cfg["fit_seed"] - 1)
                warm_codes = sample_output_codes(carrier, 16, warm_rng)
                warm = fit(warm_codes)
                del warm, warm_codes
            # The full coordinate fit is deterministic and is reused across the
            # three acceptance repetitions. Its measured cost is charged each time.
            (full_candidate, _, full_rhs), full_seconds = timed_cuda(lambda: fit(full_codes))
            for repeat in range(cfg["repetitions"]):
                offset = stream_offset + repeat
                fit_rng = torch.Generator(device=device).manual_seed(cfg["fit_seed"] + offset)
                accept_rng = torch.Generator(device=device).manual_seed(cfg["accept_seed"] + offset)
                nested_codes = sample_output_codes(carrier, max(budgets), fit_rng)
                acceptance = sample_probes(hidden, learning.validation_probes, accept_rng, orthogonal=True)
                accept_response, accept_seconds = timed_cuda(lambda: chunked_vector_responses(
                    suffix, hidden, acceptance, learning.probe_epsilon, learning.output_query_chunk))
                # Rotate the sampled-budget order to avoid a systematic timing order.
                offset_order = (batch_index + boundary + repeat) % len(budgets)
                order = budgets[offset_order:] + budgets[:offset_order]
                for budget in ["full"] + order:
                    k = model_cfg.num_classes if budget == "full" else budget
                    if budget == "full":
                        candidate, rhs, fit_seconds = full_candidate, full_rhs, full_seconds
                    else:
                        (candidate, _, rhs), fit_seconds = timed_cuda(lambda: fit(nested_codes[:, :k]))
                    (selected, accepted), select_seconds = timed_cuda(lambda: select_operator(
                        carrier, candidate, acceptance, accept_response))
                    mse = (operator_response(selected, heldout) - heldout_response).square().mean((1, 2))
                    relative = mse / carrier_mse
                    if not bool(torch.isfinite(relative).all()):
                        raise RuntimeError("Nonfinite response metric")
                    rows.append({
                        "batch": batch_index, "boundary": boundary, "repeat": repeat,
                        "budget_label": str(budget), "output_codes": k, "samples": len(images),
                        "dataset_indices": cohort[processed:processed + len(images)],
                        "dictionary_modes": len(names),
                        "logical_suffix_queries_per_image": 2 * (len(names) * k + learning.validation_probes),
                        "relative_mse_per_image": relative.tolist(),
                        "carrier_mse_per_image": carrier_mse.tolist(), "mse_per_image": mse.tolist(),
                        "accepted": accepted.tolist(),
                        "relative_rhs_difference_per_image": ((rhs-full_rhs).norm(dim=1)
                            / full_rhs.norm(dim=1).clamp_min(1e-20)).tolist(),
                        "fit_seconds": fit_seconds,
                        "fit_accept_seconds": fit_seconds + accept_seconds + select_seconds,
                    })
                    del candidate, rhs, selected, accepted
                del nested_codes, acceptance, accept_response
            del full_candidate, full_rhs, full_codes, carrier, dictionary, gram, heldout, heldout_response
        processed += len(images)
        progress = {"dataset": cfg["dataset"], "seed": cfg["seed"], "processed": processed,
                    "total": cfg["samples"], "elapsed_seconds": time.perf_counter() - start}
        print(json.dumps(progress), flush=True)
        destination.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    if processed != cfg["samples"]:
        raise RuntimeError("Incomplete image cohort")
    expected_rows = ((processed + cfg["batch_size"] - 1) // cfg["batch_size"]
                     * (model_cfg.depth - 1) * cfg["repetitions"] * (len(budgets) + 1))
    if len(rows) != expected_rows:
        raise RuntimeError("Incomplete budget/repetition/boundary grid")
    after = tensor_digest({"model": model.state_dict(), "mirrors": mirrors.state_dict()})
    if before != after:
        raise RuntimeError("Frozen state changed")
    if sha(cfg["checkpoint"]) != model_checkpoint_sha:
        raise RuntimeError("Checkpoint file changed during diagnostic")
    source_paths = [Path(__file__), ROOT / "scripts/audit_frozen_route_swap.py"]
    source_paths += [ROOT / "src" / p for p in (
        "methods/component_postln/operators.py", "methods/component_postln/feedback.py",
        "methods/component_scaling/feedback.py", "methods/component_scaling_fast/feedback.py",
        "methods/transformer_ebd/component_route_operator.py",
        "methods/transformer_ebd/forward_response_credit.py")]
    report = {
        "status": "completed", "config": cfg, "dataset": cfg["dataset"], "seed": cfg["seed"],
        "checkpoint_epoch": state["epoch"], "checkpoint_sha256": model_checkpoint_sha,
        "config_sha256": sha(config_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": model_cfg.to_dict(), "method": "frozen_FOCF_output_budget", "learning": asdict(learning),
        "data": data_info, "cohort_indices": cohort, "sample_count": processed,
        "train_accuracy": None, "test_accuracy": None,
        "diagnostic_accuracy": correct / processed, "diagnostic_loss": loss_sum / processed,
        "frozen_state_before": before, "frozen_state_after": after,
        "maximum_suffix_relative_error": suffix_relative_error,
        "wall_clock_seconds": time.perf_counter() - start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "concurrency_policy": cfg.get("concurrency_policy", "Run one diagnostic process on an otherwise idle GPU"),
        "protocol": {
            "parameter_updates": 0, "mirror_updates": 0, "autograd_used": False,
            "full_class_space_dictionary_and_gram": True,
            "nested_random_orthogonal_output_codes": True,
            "acceptance_probes_shared_across_budgets": True,
            "evaluation_probes_shared_across_budgets_and_repetitions": True,
            "fit_acceptance_evaluation_rngs_independent": True,
            "labels_used_for_fitting_or_selection": False,
            "complete_basis_fit_reused_across_acceptance_repetitions": True,
            "aggregation": "Equal image, repetition and fitted-boundary weight within each model seed",
            "timing": "Synchronized CUDA wall time for coefficient fit, acceptance responses and selection; frozen dictionary/Gram, cached forward states and evaluation probes excluded; full-fit measured cost charged for every repetition",
            "cohort": "First prespecified images in the checkpoint's existing internal evaluation cohort",
        },
        "metrics": summarize_rows(rows), "rows": rows,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
