#!/usr/bin/env python3
"""Measure sample-dependent downstream routing in existing GrAPE models.

No FOCF operator, fitted shared baseline, or learning-rule modification is used.
All derivatives are offline measurements of the frozen baseline's own suffix.
The centered logit Jacobian removes the irrelevant common-logit direction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.trainer import build_loaders


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(tensors):
    h = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def suffix_function(model, boundary):
    def suffix(hidden):
        for block in model.blocks[boundary + 1:]:
            hidden, _ = block(hidden)
        return model.head(model.final_norm(hidden)[:, 0])
    return suffix


def exact_operator(suffix, hidden):
    with torch.enable_grad():
        leaf = hidden.detach().requires_grad_(True)
        logits = suffix(leaf)
        centered = logits - logits.mean(-1, keepdim=True)
        result = [torch.autograd.grad(centered[:, c].sum(), leaf,
                  retain_graph=c + 1 < logits.shape[-1])[0].detach()
                  for c in range(logits.shape[-1])]
    return torch.stack(result, 1)


@torch.no_grad()
def directional_effect(suffix, hidden, labels, credit, reference, epsilon):
    # Same recipient-state displacement norm in all arms. The donor operator
    # acts on the recipient's error, not on the donor's label/error.
    unit = credit / credit.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-20)
    step = epsilon * hidden.flatten(1).norm(dim=1)
    minus = F.cross_entropy(suffix(hidden - step[:, None, None] * unit).double(),
                            labels, reduction="none")
    plus = F.cross_entropy(suffix(hidden + step[:, None, None] * unit).double(),
                           labels, reduction="none")
    reference_norm = reference.flatten(1).norm(dim=1)
    effect = (plus - minus) / (2 * step * reference_norm).clamp_min(1e-20)
    cosine = F.cosine_similarity(credit.flatten(1), reference.flatten(1), dim=1)
    return effect, cosine, (plus - minus > 0).double()


def run(cfg, job, device, smoke):
    start = time.perf_counter()
    raw = Path(job["checkpoint"]).read_bytes()
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    assert state["epoch"] == 100
    protocol = state["protocol"]
    assert protocol["norm_layout"] == "post" and protocol["method"] == "grape"
    assert protocol["seed"] == job["seed"]
    expected_anchors = 0 if job["method"] == "grape" else 10
    assert state["learner"]["calibration_batches"] == expected_anchors
    model_cfg = ViTConfig(image_size=32, patch_size=4, num_classes=10,
                          embed_dim=128, depth=4, num_heads=4, ffn_dim=256)
    model = AuditedViT(model_cfg, "post").to(device).eval()
    model.load_state_dict(state["learner"]["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    feedback = [a.to(device) for a in state["learner"]["feedback"]]
    frozen = dict(model.state_dict())
    frozen.update({f"feedback_{i}": a for i, a in enumerate(feedback)})
    before = digest(frozen)
    args = type("Args", (), dict(dataset="cifar10", data_root="cifar10_data",
        augmentation=False, validation_samples=1000, train_subset_samples=5000,
        validation_split_seed=2027, seed=job["seed"], batch_size=cfg["batch_size"],
        num_workers=0, device=str(device), evaluate_test_final=False))()
    loaders = build_loaders(args)
    dataset = loaders[2].dataset
    count = cfg["batch_size"] if smoke else cfg["validation_examples"]
    raw_indices = [int(i) for i in dataset.indices[:count]]
    labels_all = [int(dataset.dataset.targets[i]) for i in raw_indices]
    example_indices, seen = [], set()
    for i, label in enumerate(labels_all):
        if label not in seen:
            example_indices.append(i)
            seen.add(label)
        if len(example_indices) == 4:
            break
    assert len(example_indices) == 4
    arrays = {"images": np.stack([dataset.dataset.data[raw_indices[i]] for i in example_indices]),
              "example_indices": np.asarray(example_indices), "validation_indices": np.asarray(raw_indices)}
    records = []
    validation_correct = 0
    validation_loss = 0.
    for boundary in cfg["boundaries"]:
        suffix = suffix_function(model, boundary)
        b = feedback[boundary].reshape(10, 65, 128)
        b = b - b.mean(0, keepdim=True)
        sums = torch.zeros_like(b, dtype=torch.float64)
        normalized_sum = torch.zeros_like(sums)
        energy = 0.
        per_image = {name: [] for name in ["grape_effect", "exchanged_effect", "matched_effect",
            "grape_cosine", "exchanged_cosine", "matched_cosine",
            "grape_descends", "exchanged_descends", "matched_descends"]}
        max_fd_error, max_credit_error, max_suffix_error = 0., 0., 0.
        example_operators = {}
        offset = 0
        for x, y in loaders[2]:
            if offset >= count:
                break
            length = min(len(y), count - offset)
            x, y = x[:length].to(device), y[:length].to(device)
            assert length > 1
            with torch.no_grad():
                cache = model.forward_with_cache(x)
                hidden = cache.block_caches[boundary].block_output
                max_suffix_error = max(max_suffix_error, float((suffix(hidden) - cache.logits).abs().max()))
                error = cache.logits.softmax(-1) - F.one_hot(y, 10).to(cache.logits)
                if boundary == cfg["boundaries"][0]:
                    validation_correct += int((cache.logits.argmax(-1) == y).sum())
                    validation_loss += float(F.cross_entropy(cache.logits, y, reduction="sum"))
            j = exact_operator(suffix, hidden)
            with torch.no_grad():
                norms = j.flatten(1).norm(dim=1)
                sums += j.double().sum(0)
                normalized_sum += (j / norms[:, None, None, None].clamp_min(1e-20)).double().sum(0)
                energy += float(j.double().square().sum())
                credits = {"grape": torch.einsum("bc,cnd->bnd", error, b),
                           "matched": torch.einsum("bc,bcnd->bnd", error, j),
                           "exchanged": torch.einsum("bc,bcnd->bnd", error, j.roll(1, 0))}
            # Independent differentiation verifies that centering and class
            # contraction recover the actual CE credit of this same model.
            with torch.enable_grad():
                leaf = hidden.detach().requires_grad_(True)
                direct = torch.autograd.grad(F.cross_entropy(suffix(leaf), y, reduction="sum"), leaf)[0]
            rel = (direct - credits["matched"]).norm() / direct.norm().clamp_min(1e-20)
            max_credit_error = max(max_credit_error, float(rel))
            for name, credit in credits.items():
                effect, cosine, descends = directional_effect(suffix, hidden, y, credit,
                    credits["matched"], cfg["hidden_relative_step"])
                max_fd_error = max(max_fd_error, float((effect - cosine.double()).abs().max()))
                for key, value in [("effect", effect), ("cosine", cosine), ("descends", descends)]:
                    per_image[f"{name}_{key}"].extend(value.cpu().tolist())
            for index in example_indices:
                if offset <= index < offset + length:
                    example_operators[index] = j[index - offset].cpu().numpy()
            offset += length
        assert offset == count and len(example_operators) == 4
        assert max_credit_error < 1e-5 and max_suffix_error < 1e-5
        # The two terms in E||J-B||^2 = E||J-EJ||^2 + ||EJ-B||^2
        # are reported as descriptive operator geometry, not accuracy shares.
        mean = sums / count
        variation = energy / count - float(mean.square().sum())
        directional_variation = 1 - float((normalized_sum / count).square().sum())
        baseline = b.double()
        scale = float((mean * baseline).sum() / baseline.square().sum().clamp_min(1e-20))
        mismatch = float((mean - scale * baseline).square().sum())
        total = energy / count
        record = {"boundary": boundary, "samples": count,
            "input_variation_energy_fraction": variation / total,
            "unit_operator_directional_variation": directional_variation,
            "scale_aligned_shared_mismatch_fraction": mismatch / total,
            "scale_aligned_total_error_fraction": (variation + mismatch) / total,
            "diagnostic_global_scale": scale,
            "max_fd_cosine_absolute_error": max_fd_error,
            "max_exact_credit_relative_error": max_credit_error,
            "max_suffix_logit_absolute_error": max_suffix_error,
            **{k: float(np.mean(v)) for k, v in per_image.items()}}
        for k, values in per_image.items():
            arrays[f"boundary{boundary}_{k}"] = np.asarray(values)
        arrays[f"boundary{boundary}_example_operators"] = np.stack([example_operators[i] for i in example_indices])
        arrays[f"boundary{boundary}_shared_operator"] = b.cpu().numpy()
        records.append(record)
        print(json.dumps({"method": job["method"], "seed": job["seed"], **record}), flush=True)
    after = digest(frozen)
    assert before == after and all(p.grad is None for p in model.parameters())
    directory = ROOT / cfg["output_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / (f'{job["method"]}_seed{job["seed"]}' + ("_smoke" if smoke else ""))
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    record = {"method": job["method"], "seed": job["seed"], "config": cfg,
        "checkpoint": job["checkpoint"], "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_epoch": state["epoch"], "calibration_batches": expected_anchors,
        "script_sha256": sha(__file__), "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": model_cfg.to_dict(), "training_protocol": protocol, "data": loaders[-1],
        "test_accessed": False, "parameter_updates": 0, "feedback_updates": 0,
        "frozen_digest_before": before, "frozen_digest_after": after,
        "validation_accuracy": validation_correct / count, "validation_loss": validation_loss / count,
        "example_classes": [dataset.dataset.classes[labels_all[i]] for i in example_indices],
        "example_selection": "First four distinct labels in the predetermined validation prefix, chosen before measuring operators",
        "records": records, "array_sha256": sha(str(prefix) + ".npz"),
        "wall_clock_seconds": time.perf_counter() - start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "scope": "Frozen baseline suffix geometry and hidden-state directional loss response, not a training intervention or a parameter-optimizer comparison"}
    Path(str(prefix) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/diagnostics/routing.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    device = torch.device(args.device)
    source = json.loads((ROOT / cfg["checkpoint_report"]).read_text())
    jobs = sorted([j for j in source["runs"] if j["method"] in cfg["methods"]
                  and j["seed"] in cfg["seeds"]], key=lambda j: (j["method"], j["seed"]))
    assert len(jobs) == len(cfg["methods"]) * len(cfg["seeds"])
    if args.smoke:
        jobs = jobs[:1]
    results = [run(cfg, job, device, args.smoke) for job in jobs]
    out = ROOT / cfg["output_directory"] / ("summary_smoke.json" if args.smoke else "summary.json")
    out.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg, "source_report_sha256": sha(ROOT / cfg["checkpoint_report"]),
        "results": results}, indent=2) + "\n")
    print("Saved " + str(out), flush=True)


if __name__ == "__main__":
    main()
