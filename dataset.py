import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


SEED = 42
VAL_RATIO = 0.20

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

PROJECT_ROOT = Path(__file__).resolve().parent

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"


def read_annotations(filename):

    annotation_path = DATA_ROOT / "annotations" / filename

    image_names = []
    labels = []

    with open(annotation_path, "r") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 4:
                continue

            image_name = parts[0]

            class_id = int(parts[1]) - 1

            image_path = DATA_ROOT / "images" / f"{image_name}.jpg"

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

            image_names.append(image_name)
            labels.append(class_id)

    return np.array(image_names), np.array(labels)


def create_splits():

    SPLIT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    trainval_names, trainval_labels = read_annotations(
        "trainval.txt"
    )

    test_names, test_labels = read_annotations(
        "test.txt"
    )

    indices = np.arange(len(trainval_names))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_RATIO,
        stratify=trainval_labels,
        random_state=SEED
    )

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    train_names = trainval_names[train_idx]
    val_names = trainval_names[val_idx]

    train_labels = trainval_labels[train_idx]
    val_labels = trainval_labels[val_idx]

    assert len(set(train_names) & set(val_names)) == 0
    assert len(set(train_names) & set(test_names)) == 0
    assert len(set(val_names) & set(test_names)) == 0

    assert len(np.unique(trainval_labels)) == 37
    assert len(np.unique(test_labels)) == 37

    np.save(SPLIT_ROOT / "train_indices.npy", train_idx)
    np.save(SPLIT_ROOT / "val_indices.npy", val_idx)

    np.save(
        SPLIT_ROOT / "test_indices.npy",
        np.arange(len(test_names))
    )

    np.save(SPLIT_ROOT / "train_names.npy", train_names)
    np.save(SPLIT_ROOT / "val_names.npy", val_names)
    np.save(SPLIT_ROOT / "test_names.npy", test_names)

    np.save(SPLIT_ROOT / "train_labels.npy", train_labels)
    np.save(SPLIT_ROOT / "val_labels.npy", val_labels)
    np.save(SPLIT_ROOT / "test_labels.npy", test_labels)

    metadata = {
        "dataset": "Oxford-IIIT Pet",
        "seed": SEED,
        "validation_ratio": VAL_RATIO,
        "train_size": len(train_names),
        "val_size": len(val_names),
        "test_size": len(test_names),
        "num_classes": 37,
        "data_root": str(DATA_ROOT)
    }

    with open(SPLIT_ROOT / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 50)

    print(f"TRAIN: {len(train_names)} images")
    print(f"VAL:   {len(val_names)} images")
    print(f"TEST:  {len(test_names)} images")

    print(f"\nNumber of classes: {len(np.unique(trainval_labels))}")

    print("\nTraining distribution:")

    train_counts = Counter(train_labels)

    for class_id in sorted(train_counts):
        print(f"Class {class_id:02d}: {train_counts[class_id]} images")

    print("\nAll splits saved successfully!")


if __name__ == "__main__":
    create_splits()