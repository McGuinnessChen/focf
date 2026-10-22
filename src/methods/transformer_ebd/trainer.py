"""Dataset transforms, fixed splits and loaders used in the paper."""


from __future__ import annotations


import argparse


from pathlib import Path


import numpy as np


import torch


from torch.utils.data import DataLoader, Subset


from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]


from methods.transformer_ebd.tiny_imagenet import TinyImageNet


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


_DATASET_SPECS = {
    "cifar10": (
        datasets.CIFAR10,
        10,
        "cifar10_data",
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
    "cifar100": (
        datasets.CIFAR100,
        100,
        "cifar100_data",
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
    "tiny_imagenet": (
        TinyImageNet,
        200,
        "tiny_imagenet_data",
        (0.4850, 0.4560, 0.4060),
        (0.2290, 0.2240, 0.2250),
    ),
}


def _dataset_spec(dataset_name: str):
    try:
        return _DATASET_SPECS[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from error


def _transforms(dataset_name: str, augmentation: bool):
    _, _, _, mean, std = _dataset_spec(dataset_name)
    image_size = 64 if dataset_name == "tiny_imagenet" else 32
    normalize = transforms.Normalize(mean=mean, std=std)
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if not augmentation:
        return eval_transform, eval_transform
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace):
    dataset_class, num_classes, default_root, _, _ = _dataset_spec(args.dataset)
    train_transform, eval_transform = _transforms(args.dataset, args.augmentation)
    data_root_name = args.data_root if args.data_root is not None else default_root
    data_root = (PROJECT_ROOT / data_root_name).resolve()
    train_augmented = dataset_class(
        str(data_root), train=True, transform=train_transform, download=False
    )
    train_evaluation = dataset_class(
        str(data_root), train=True, transform=eval_transform, download=False
    )
    if args.validation_samples <= 0 or args.validation_samples >= len(train_augmented):
        raise ValueError("validation_samples must be between zero and the train-set size")
    split_generator = torch.Generator().manual_seed(args.validation_split_seed)
    permutation = torch.randperm(len(train_augmented), generator=split_generator).tolist()
    validation_indices = permutation[: args.validation_samples]
    training_indices = permutation[args.validation_samples :]
    if args.train_subset_samples > 0:
        training_indices = training_indices[: args.train_subset_samples]
    if len(training_indices) < args.batch_size:
        raise ValueError("Training subset must contain at least one full batch")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_augmented, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        generator=loader_generator,
    )
    train_eval_loader = DataLoader(
        Subset(train_evaluation, training_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    validation_loader = DataLoader(
        Subset(train_evaluation, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    test_loader = None
    if args.evaluate_test_final:
        test_dataset = dataset_class(
            str(data_root), train=False, transform=eval_transform, download=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
        )
    split_bytes = np.asarray(validation_indices, dtype=np.int64).tobytes()
    split_sha = __import__("hashlib").sha256(split_bytes).hexdigest()
    return train_loader, train_eval_loader, validation_loader, test_loader, {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "data_root": str(data_root),
        "training_samples": len(training_indices),
        "validation_samples": len(validation_indices),
        "validation_seed": args.validation_split_seed,
        "validation_indices_sha256": split_sha,
        "augmentation": bool(args.augmentation),
        "test_access_policy": "final_once" if args.evaluate_test_final else "never",
    }
