import os
import random

import numpy as np
import torch


def configure_reproducibility(seed, deterministic=True):
    """Configure process-level RNGs and deterministic CUDA behavior."""
    seed = int(seed)
    random.seed(seed)
    # NumPy's legacy RNG requires an unsigned 32-bit seed. Preserve the
    # existing nonnegative-seed behavior and PyTorch's native signed seeds.
    numpy_seed = seed % (2**32) if seed < 0 else seed
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)

    return {
        "seed": seed,
        "numpy_seed": numpy_seed,
        "deterministic_algorithms": deterministic,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
