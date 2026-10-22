import json
import hashlib
import importlib.metadata
import math
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Subset


def make_json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def as_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() != 1:
            raise ValueError("Expected scalar metric tensor")
        return float(value.item())
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def metrics_are_invalid(*values):
    for value in values:
        numeric = as_float(value)
        if numeric is not None and not math.isfinite(numeric):
            return True
    return False


def make_smoke_subset(dataset, requested_samples, batch_size):
    sample_count = max(batch_size * 2, int(requested_samples))
    sample_count = min(len(dataset), sample_count)
    sample_count = sample_count - (sample_count % batch_size)
    if sample_count <= 0:
        sample_count = min(len(dataset), batch_size)
    return Subset(dataset, list(range(sample_count)))


def _git_output(repo_root, *arguments):
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo_root,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def get_git_provenance(repo_root):
    """Return provenance only when repo_root is the repository top-level."""
    repo_root = Path(repo_root).resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"Repository root does not exist: {repo_root}")

    try:
        git_top_level = Path(
            _git_output(repo_root, "rev-parse", "--show-toplevel")
        ).resolve()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Cannot identify a Git repository at {repo_root}") from exc

    if git_top_level != repo_root:
        raise RuntimeError(
            "repo_root must be the Git top-level: "
            f"received {repo_root}, Git reported {git_top_level}"
        )

    try:
        commit = _git_output(repo_root, "rev-parse", "--verify", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    try:
        status = _git_output(
            repo_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        dirty = bool(status)
    except (OSError, subprocess.CalledProcessError):
        dirty = None

    return {
        "top_level": str(git_top_level),
        "commit": commit or "unknown",
        "dirty": dirty,
    }


def get_git_commit_hash(repo_root):
    """Compatibility helper; strict validation is performed by ExperimentLogger."""
    try:
        return get_git_provenance(repo_root)["commit"]
    except RuntimeError:
        return "unknown"


def get_environment_fingerprint(repo_root=None):
    cuda_available = bool(torch.cuda.is_available())
    gpus = []
    if cuda_available:
        try:
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0
        for index in range(gpu_count):
            gpu = {"index": index}
            try:
                properties = torch.cuda.get_device_properties(index)
                gpu.update(
                    {
                        "name": properties.name,
                        "total_memory_mb": properties.total_memory / (1024 * 1024),
                        "compute_capability": [properties.major, properties.minor],
                    }
                )
            except Exception as exc:
                gpu["query_error"] = str(exc)
            gpus.append(gpu)

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    packages = sorted(
        f"{distribution.metadata.get('Name') or 'unknown'}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    package_bytes = ("\n".join(packages) + "\n").encode("utf-8")
    try:
        driver_lines = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).splitlines()
        driver_versions = sorted(set(line.strip() for line in driver_lines if line.strip()))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        driver_versions = []

    repository_files = {}
    if repo_root is not None:
        for relative_path in ("requirements.txt", "configs/datasets/local_datasets.json"):
            path = Path(repo_root) / relative_path
            if path.is_file():
                repository_files[relative_path] = _sha256_file(path)

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "nvidia_driver_versions": driver_versions,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "python_packages": packages,
        "python_packages_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "repository_file_sha256": repository_files,
    }


def get_orchestration_context():
    keys = {
        "launch_id": "EBD_LAUNCH_ID",
        "job_id": "EBD_JOB_ID",
        "config_sha256": "EBD_CONFIG_SHA256",
        "expected_git_commit": "EBD_EXPECTED_GIT_COMMIT",
        "launch_manifest": "EBD_LAUNCH_MANIFEST",
    }
    return {name: os.environ.get(variable) for name, variable in keys.items()}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_dataset_provenance(repo_root, dataset):
    manifest_path = Path(repo_root) / "configs" / "datasets" / "local_datasets.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Dataset provenance manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        entry = manifest["datasets"][str(dataset)]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Dataset is not approved in the provenance manifest: {dataset}") from exc

    validated_files = []
    for expected in entry.get("files", []):
        relative_path = Path(expected["path"])
        path = (Path(repo_root) / relative_path).resolve()
        try:
            path.relative_to(Path(repo_root).resolve())
        except ValueError as exc:
            raise RuntimeError(f"Dataset path escapes repository root: {relative_path}") from exc
        if not path.is_file():
            raise RuntimeError(f"Dataset file is missing: {relative_path}")
        actual_size = path.stat().st_size
        if actual_size != int(expected["size"]):
            raise RuntimeError(
                f"Dataset size mismatch for {relative_path}: "
                f"expected {expected['size']}, found {actual_size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"Dataset SHA-256 mismatch for {relative_path}: "
                f"expected {expected['sha256']}, found {actual_sha256}"
            )
        validated_files.append(
            {
                "path": relative_path.as_posix(),
                "size": actual_size,
                "sha256": actual_sha256,
            }
        )
    if not validated_files:
        raise RuntimeError(f"Dataset provenance has no files: {dataset}")
    return {
        "dataset": str(dataset),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "validated": True,
        "files": validated_files,
    }


def _safe_run_component(value):
    safe_value = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value)
    )
    return safe_value or "unnamed"


class ExperimentLogger:
    def __init__(
        self,
        repo_root,
        dataset,
        model,
        method,
        seed,
        config,
        overwrite=True,
        run_tag=None,
        require_clean_git=True,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.dataset = dataset
        self.model = model
        self.method = method
        self.seed = seed
        self.run_tag = run_tag

        git = get_git_provenance(self.repo_root)
        self.git_commit_hash = git["commit"]
        self.git_dirty = git["dirty"]
        if require_clean_git and self.git_commit_hash == "unknown":
            raise RuntimeError(
                "Experiment logging requires a repository with a valid Git commit; "
                "pass require_clean_git=False only for an explicitly non-reproducible test"
            )
        if require_clean_git and self.git_dirty is not False:
            state = "dirty" if self.git_dirty else "of unknown cleanliness"
            raise RuntimeError(
                f"Experiment logging requires a clean Git worktree (worktree is {state}); "
                "pass require_clean_git=False only for an explicitly non-reproducible test"
            )
        self.dataset_provenance = get_dataset_provenance(self.repo_root, dataset)

        created_at = datetime.now(timezone.utc)
        date = created_at.strftime("%Y-%m-%d")
        run_name_parts = [dataset, model, method, f"seed{seed}"]
        if run_tag:
            run_name_parts.append(run_tag)
        run_name = "_".join(_safe_run_component(value) for value in run_name_parts)

        # UUID plus exclusive directory creation prevents concurrent launches from
        # ever sharing or truncating a prior run, irrespective of overwrite.
        runs_root = self.repo_root / "runs" / date
        runs_root.mkdir(parents=True, exist_ok=True)
        while True:
            self.run_id = (
                f"{created_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex}"
            )
            self.run_dir = runs_root / f"{run_name}_{self.run_id}"
            try:
                self.run_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=False)
        self.start_time = time.time()
        self.environment = get_environment_fingerprint(self.repo_root)
        self.orchestration = get_orchestration_context()

        # Keep the legacy argument in the public API, but never use it to truncate.
        self.overwrite_requested = bool(overwrite)
        self.metrics_path.touch(exist_ok=False)
        full_config = {
            "run_id": self.run_id,
            "dataset": dataset,
            "model": model,
            "method": method,
            "seed": seed,
            "run_tag": run_tag,
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "git_top_level": ".",
            "created_at": created_at.isoformat(timespec="seconds"),
            "environment": self.environment,
            "orchestration": self.orchestration,
            "artifacts_dir": str(self.artifacts_dir),
            "dataset_provenance": self.dataset_provenance,
            "config": make_json_safe(config),
        }
        with self.config_path.open("x", encoding="utf-8") as handle:
            json.dump(make_json_safe(full_config), handle, indent=2, sort_keys=True)
        self._write_orchestration_report()

    def _write_orchestration_report(self):
        report_path = os.environ.get("EBD_RUN_REPORT_PATH")
        if not report_path:
            return
        path = Path(report_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config_path": str(self.config_path),
            "metrics_path": str(self.metrics_path),
            "artifacts_dir": str(self.artifacts_dir),
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "orchestration": self.orchestration,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(make_json_safe(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def reset_peak_cuda_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_cuda_memory_mb(self):
        """Return peak allocated memory under the legacy metric name."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def peak_cuda_memory_reserved_mb(self):
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_reserved() / (1024 * 1024)

    def log_epoch(
        self,
        epoch,
        train_loss,
        test_loss,
        train_accuracy,
        test_accuracy,
        epoch_time_sec,
        learning_rate,
        failed=False,
        failure_reason=None,
        extra=None,
    ):
        peak_allocated_mb = self.peak_cuda_memory_mb()
        record = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "seed": self.seed,
            "run_tag": self.run_tag,
            "epoch": epoch,
            "train_loss": as_float(train_loss),
            "test_loss": as_float(test_loss),
            "train_accuracy": as_float(train_accuracy),
            "test_accuracy": as_float(test_accuracy),
            "epoch_time_sec": as_float(epoch_time_sec),
            "total_time_sec": time.time() - self.start_time,
            "peak_cuda_memory_mb": peak_allocated_mb,
            "peak_cuda_memory_allocated_mb": peak_allocated_mb,
            "peak_cuda_memory_reserved_mb": self.peak_cuda_memory_reserved_mb(),
            "learning_rate": as_float(learning_rate),
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "config_path": str(self.config_path),
            "failed": bool(failed),
        }
        if failure_reason:
            record["failure_reason"] = failure_reason
        if extra:
            record.update(make_json_safe(extra))

        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(make_json_safe(record), sort_keys=True) + "\n")
        return record
