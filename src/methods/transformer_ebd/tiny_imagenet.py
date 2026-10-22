"""Dataset adapter for the official Tiny ImageNet directory layout."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    """Read official train and labeled validation splits without rearranging files."""

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        transform=None,
        target_transform=None,
        download: bool = False,
    ) -> None:
        if download:
            raise ValueError("Tiny ImageNet must be downloaded and verified separately")
        root_path = Path(root)
        dataset_root = root_path / "tiny-imagenet-200"
        if not dataset_root.is_dir():
            dataset_root = root_path
        wnids_path = dataset_root / "wnids.txt"
        if not wnids_path.is_file():
            raise RuntimeError(f"Tiny ImageNet metadata is missing: {wnids_path}")

        self.root = dataset_root
        self.train = bool(train)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }
        self.samples = self._training_samples() if self.train else self._validation_samples()
        if not self.samples:
            split = "train" if self.train else "validation"
            raise RuntimeError(f"Tiny ImageNet {split} split contains no images")
        self.targets = [target for _, target in self.samples]

    def _training_samples(self):
        samples = []
        for class_name in self.classes:
            image_directory = self.root / "train" / class_name / "images"
            for image_path in sorted(image_directory.glob("*.JPEG")):
                samples.append((image_path, self.class_to_idx[class_name]))
        return samples

    def _validation_samples(self):
        annotations_path = self.root / "val" / "val_annotations.txt"
        if not annotations_path.is_file():
            raise RuntimeError(
                f"Tiny ImageNet validation annotations are missing: {annotations_path}"
            )
        samples = []
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("	")
            if len(fields) < 2:
                raise RuntimeError(f"Malformed Tiny ImageNet annotation: {line!r}")
            filename, class_name = fields[:2]
            try:
                target = self.class_to_idx[class_name]
            except KeyError as error:
                raise RuntimeError(
                    f"Unknown Tiny ImageNet validation class: {class_name}"
                ) from error
            samples.append((self.root / "val" / "images" / filename, target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
