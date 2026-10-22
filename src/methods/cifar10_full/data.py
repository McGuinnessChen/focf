"""Use all 50,000 official training images, with no internal holdout."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from methods.transformer_ebd.trainer import PROJECT_ROOT, _dataset_spec, _transforms


def build_loaders(args, *, smoke_batches=0):
    if args.dataset != "cifar10" or args.validation_samples != 0 or args.train_subset_samples != 0:
        raise ValueError("Full CIFAR-10 requires dataset=cifar10, validation=0, subset=0")
    if smoke_batches and args.evaluate_test_final:
        raise ValueError("Smoke runs must not evaluate the official test set")
    dataset_class, classes, default_root, _, _ = _dataset_spec(args.dataset)
    root = (PROJECT_ROOT / (args.data_root or default_root)).resolve()
    augmented, evaluation = _transforms(args.dataset, args.augmentation)
    train_data = dataset_class(str(root), train=True, transform=augmented, download=False)
    eval_data = dataset_class(str(root), train=True, transform=evaluation, download=False)
    if len(train_data) != 50000 or len(eval_data) != 50000:
        raise ValueError("Expected the complete 50,000-image official training set")
    if np.bincount(train_data.targets, minlength=10).tolist() != [5000] * 10:
        raise ValueError("Unexpected official training class counts")
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(args.validation_split_seed)).tolist()
    if smoke_batches:
        indices = indices[:smoke_batches * args.batch_size]
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=args.device == "cuda")
    train = DataLoader(Subset(train_data, indices), shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed), **options)
    train_eval = DataLoader(Subset(eval_data, indices), shuffle=False, **options)
    # Empty explicitly: no training image is withheld and no test image is used
    # for validation, hyperparameter choice or checkpoint selection.
    validation = DataLoader(Subset(eval_data, []), shuffle=False, **options)
    test = None
    if args.evaluate_test_final:
        test_data = dataset_class(str(root), train=False, transform=evaluation, download=False)
        if len(test_data) != 10000 or np.bincount(test_data.targets, minlength=10).tolist() != [1000] * 10:
            raise ValueError("Expected the 10,000-image official test split")
        test = DataLoader(test_data, shuffle=False, **options)
    info = {
        "dataset": "cifar10", "num_classes": classes, "data_root": str(root),
        "training_samples": len(indices), "official_training_samples": 50000,
        "validation_samples": 0, "validation_seed": args.validation_split_seed,
        "validation_indices_sha256": hashlib.sha256(b"").hexdigest(),
        "augmentation": bool(args.augmentation), "test_samples": 10000 if test is not None else 0,
        "test_access_policy": "final_once" if test is not None else "never",
        "checkpoint_selection": "fixed final epoch; no validation or test selection",
        "protocol": "cifar10_official_full50k_test10k", "smoke_batches": smoke_batches,
    }
    return train, train_eval, validation, test, info
