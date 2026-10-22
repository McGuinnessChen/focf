"""Record local data coverage without changing the established random split."""
import hashlib

import numpy as np


def record_split_provenance(train, validation):
    training = np.asarray(train.dataset.indices, dtype=np.int64)
    held_out = np.asarray(validation.dataset.indices, dtype=np.int64)
    if np.intersect1d(training, held_out).size:
        raise ValueError("Training and validation indices overlap")
    dataset = train.dataset.dataset
    targets = np.asarray(dataset.targets, dtype=np.int64)
    classes = len(dataset.classes)
    counts = np.bincount(targets[training], minlength=classes)
    val_counts = np.bincount(targets[held_out], minlength=classes)
    return {
        "training_indices_sha256": hashlib.sha256(training.tobytes()).hexdigest(),
        "class_order_sha256": hashlib.sha256("\n".join(dataset.classes).encode()).hexdigest(),
        "training_class_counts": counts.tolist(),
        "validation_class_counts": val_counts.tolist(),
        "training_classes_present": int((counts > 0).sum()),
        "validation_classes_present": int((val_counts > 0).sum()),
        "split_strategy": "preserved random permutation, not stratified",
        "training_validation_disjoint": True,
    }
