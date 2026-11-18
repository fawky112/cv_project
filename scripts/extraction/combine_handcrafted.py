import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "features" / "handcrafted"

FEATURE_SETS = [
    "color",
    "lbp",
    "glcm",
    "hog",
    "pooled_hog"
]

EXPECTED_DIMS = {
    "color": 88,
    "lbp": 1003,
    "glcm": 40,
    "hog": 8100,
    "pooled_hog": 144
}


def combine_split(split):

    matrices = []

    reference_labels = None

    for feature_set in FEATURE_SETS:

        feature_path = (
            FEATURE_ROOT / f"{feature_set}_{split}.npy"
        )

        label_path = (
            FEATURE_ROOT / f"{feature_set}_{split}_labels.npy"
        )

        X = np.load(feature_path)

        y = np.load(label_path)

        if X.shape[1] != EXPECTED_DIMS[feature_set]:
            raise ValueError(
                f"Unexpected feature count for {feature_set}"
            )

        if reference_labels is None:
            reference_labels = y

        else:
            np.testing.assert_array_equal(
                reference_labels,
                y
            )

        matrices.append(X)

        print(
            f"{feature_set:12s}: {X.shape}"
        )

    combined = np.concatenate(
        matrices,
        axis=1
    ).astype(np.float32)

    assert combined.shape[1] == 9375

    assert np.all(np.isfinite(combined))

    np.save(
        FEATURE_ROOT / f"handcrafted_{split}.npy",
        combined
    )

    np.save(
        FEATURE_ROOT / f"handcrafted_{split}_labels.npy",
        reference_labels
    )

    print(
        f"\nCOMBINED {split.upper()}: {combined.shape}"
    )


def main():

    for split in ["train", "val"]:
        print(f"\nCombining {split.upper()} Features")
        print("-" * 50)
        combine_split(split)

    print("\nALL HANDCRAFTED FEATURES COMBINED SUCCESSFULLY")


if __name__ == "__main__":
    main()
