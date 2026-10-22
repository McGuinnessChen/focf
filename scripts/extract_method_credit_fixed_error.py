#!/usr/bin/env python3
"""Compare each trained method's feedback under one common output contrast.

BP, GrAPE and FOCF retain their own trained model/feedback states. This is a
qualitative cross-model comparison, not a common-Jacobian approximation test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_frozen_route_swap import tensor_digest
from methods.component_postln.feedback import ForwardMirrorBank
from methods.component_postln.learner import CommonConfig
from methods.component_postln.operators import downstream_component_dictionary
from methods.grape_vit_audit.model import AuditedViT
from methods.transformer_ebd.component_route_operator import sample_output_codes, select_operator, trace_component_operator
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.forward_response_credit import sample_probes, forward_vector_responses
from methods.transformer_ebd.trainer import build_loaders


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loader_args = type("Args", (), dict(dataset="cifar10", data_root="cifar10_data",
        augmentation=False, validation_samples=1000, train_subset_samples=5000,
        validation_split_seed=2027, seed=cfg["seed"], batch_size=cfg["batch_size"],
        num_workers=0, device="cuda", evaluate_test_final=False))()
    loaders = build_loaders(loader_args)
    validation = loaders[2]
    images, labels = next(iter(validation))
    with np.load(ROOT / cfg["baseline_arrays"]) as baseline:
        selected = baseline["example_indices"].copy()
        raw_ids = baseline["validation_indices"][selected].copy()
        originals = baseline["images"].copy()
        saved_shared = baseline["boundary0_shared_operator"].copy()
    current_ids = np.array([validation.dataset.indices[int(i)] for i in selected])
    np.testing.assert_array_equal(current_ids, raw_ids)
    np.testing.assert_array_equal(originals, validation.dataset.dataset.data[raw_ids])
    classes = [validation.dataset.dataset.classes[int(labels[int(i)])] for i in selected]
    arrays = dict(images=originals, dataset_indices=raw_ids, selected_batch_indices=selected)
    contrast = torch.full((10,), -.1, device="cuda")
    contrast[cfg["output_contrast_class"]] += 1
    arrays["output_contrast"] = contrast.cpu().numpy()
    checks, records = {}, {}
    model_cfg = ViTConfig(image_size=32, patch_size=4, num_classes=10,
                         embed_dim=128, depth=4, num_heads=4, ffn_dim=256)
    for method in ("bp", "grape", "component"):
        state = torch.load(cfg["checkpoints"][method], map_location="cpu")
        assert state["epoch"] == 100 and state["protocol"]["seed"] == cfg["seed"]
        assert state["protocol"]["method"] == method
        model = AuditedViT(model_cfg, "post").cuda().float().eval()
        model.load_state_dict(state["learner"]["model"], strict=True)
        model.requires_grad_(False)
        boundary = cfg["boundary"]
        mirrors, saved_feedback = None, None
        if method == "component":
            learning = CommonConfig(**state["learner"]["config"])
            mirrors = ForwardMirrorBank(model, learning)
            mirrors.load_state_dict(state["learner"]["component"]["mirrors"])
        if method == "grape":
            saved_feedback = state["learner"]["feedback"][boundary].cuda()

        def frozen_state():
            return {"model": model.state_dict(),
                    "mirrors": mirrors.state_dict() if mirrors else None,
                    "feedback": saved_feedback}

        before = tensor_digest(frozen_state())

        def suffix(tokens):
            for block in model.blocks[boundary + 1:]:
                tokens, _ = block(tokens)
            return model.head(model.final_norm(tokens)[:, 0])

        with torch.no_grad():
            cache = model.forward_with_cache(images.cuda())
            hidden = cache.block_caches[boundary].block_output
        if method == "bp":
            with torch.enable_grad():
                leaf = hidden.detach().requires_grad_(True)
                credit = torch.autograd.grad((suffix(leaf) * contrast).sum(), leaf)[0].detach()
            # Check the contraction against independent per-class derivatives.
            with torch.enable_grad():
                leaf = hidden[selected].detach().requires_grad_(True)
                logits = suffix(leaf)
                columns = [torch.autograd.grad(logits[:, c].sum(), leaf,
                    retain_graph=c < 9)[0].detach() for c in range(10)]
            independent = torch.einsum("c,bcnd->bnd", contrast, torch.stack(columns, 1))
            relative = float((independent - credit[selected]).norm() / credit[selected].norm())
            assert relative < 1e-5
            checks["bp_contrast_contraction_relative_error"] = relative
        elif method == "grape":
            with torch.no_grad():
                operator = saved_feedback.reshape(10, 65, 128)
                shared = torch.einsum("c,cnd->nd", contrast, operator)
                credit = shared[None].expand(len(images), -1, -1)
                old = torch.einsum("c,cnd->nd", contrast, torch.as_tensor(saved_shared, device="cuda"))
                relative = float((old - shared).norm() / shared.norm())
                assert relative < 1e-5
                checks["grape_previous_figure_relative_error"] = relative
        else:
            with torch.no_grad():
                carrier, dictionary, names = downstream_component_dictionary(model, cache.block_caches, mirrors.maps, boundary)
                rng = torch.Generator(device="cuda").manual_seed(cfg["probe_seed"])
                codes = sample_output_codes(carrier, learning.output_probes or 10, rng)
                candidate, _, _ = trace_component_operator(suffix, hidden, carrier, dictionary, codes,
                    epsilon=learning.probe_epsilon, ridge=learning.operator_ridge,
                    trust_ratio=learning.trust_ratio, batch_output_queries=True)
                probes = sample_probes(hidden, learning.validation_probes, rng, orthogonal=True)
                response = forward_vector_responses(suffix, hidden, probes, learning.probe_epsilon)
                operator, accepted = select_operator(carrier, candidate, probes, response)
                credit = torch.einsum("c,bcnd->bnd", contrast, operator)
                checks["focf_selected_acceptance"] = accepted[selected].tolist()
                checks["focf_dictionary_modes"] = len(names)
        after = tensor_digest(frozen_state())
        assert before == after
        full = credit[selected].cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(full.reshape(len(selected), -1), axis=1)
        assert np.isfinite(full).all() and np.all(norms > 1e-20)
        unit = full / norms[:, None, None]
        key = "focf" if method == "component" else method
        arrays[key + "_credit"] = full
        arrays[key + "_unit_credit"] = unit
        arrays[key + "_patch_magnitude"] = np.linalg.norm(unit[:, 1:], axis=-1).reshape(-1, 8, 8)
        arrays[key + "_cls_fraction"] = (unit[:, 0] ** 2).sum(-1)
        np.testing.assert_allclose((unit ** 2).sum((1, 2)), 1, atol=1e-12)
        assert np.allclose(arrays[key + "_patch_magnitude"].reshape(4, -1).__pow__(2).sum(-1)
                           + arrays[key + "_cls_fraction"], 1, atol=1e-12)
        records[key] = dict(checkpoint=cfg["checkpoints"][method],
            checkpoint_sha256=sha(cfg["checkpoints"][method]), checkpoint_epoch=state["epoch"],
            learning=state["learner"]["config"], model=model_cfg.to_dict(),
            frozen_before=before, frozen_after=after, train_accuracy=None, test_accuracy=None,
            diagnostic_loss=float(torch.nn.functional.cross_entropy(cache.logits, labels.cuda())),
            cls_fraction=arrays[key + "_cls_fraction"].tolist())
        del credit, model, state, cache, hidden, mirrors, saved_feedback
    np.testing.assert_allclose(arrays["grape_unit_credit"],
                               np.repeat(arrays["grape_unit_credit"][:1], 4, axis=0), atol=0)
    destination = ROOT / cfg["output_prefix"]
    np.savez_compressed(str(destination) + ".npz", **arrays)
    record = dict(config=cfg, config_sha256=sha(args.config),
        source_sha256=sha(__file__), checkpoint_report_sha256=sha(ROOT / cfg["checkpoint_report"]),
        baseline_arrays_sha256=sha(ROOT / cfg["baseline_arrays"]),
        array_sha256=sha(Path(str(destination) + ".npz")),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        seed=cfg["seed"], dataset=loaders[-1], classes=classes,
        dataset_indices=raw_ids.tolist(), methods=records, checks=checks,
        selection="Same four prespecified input images as the previous Figure 1A",
        error="Fixed truck-minus-class-mean output contrast for every model and image",
        normalization="Whole credit vector, including CLS, normalized to unit Euclidean norm",
        color_scale="One common linear magnitude scale across all methods and images",
        method="frozen_cross_method_credit_visualization", parameter_updates=0, feedback_updates=0,
        test_accessed=False, model_seeds=1,
        wall_clock_seconds=time.perf_counter() - start,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2**20,
        concurrency_policy="Single short extraction on idle GPU; no efficiency comparison")
    Path(str(destination) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output":str(destination), "checks":checks,
        "cls_percent":{k: (100 * arrays[k + "_cls_fraction"]).tolist() for k in records}}, indent=2))


if __name__ == "__main__":
    main()
