import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

HANDCRAFTED_DIR = ROOT / "features" / "handcrafted"
TRANSFORMER_DIR = ROOT / "features" / "transformer"
BOVW_DIR = ROOT / "features" / "bovw"

OUTPUT_DIR = ROOT / "features" / "combined"


FEATURE_GROUPS = [
    {
        "name": "color",
        "directory": HANDCRAFTED_DIR,
        "prefix": "color",
        "dimension": 88,
        "dictionary": "color_feature_dictionary.json"
    },
    {
        "name": "lbp",
        "directory": HANDCRAFTED_DIR,
        "prefix": "lbp",
        "dimension": 1003,
        "dictionary": "lbp_feature_dictionary.json"
    },
    {
        "name": "glcm",
        "directory": HANDCRAFTED_DIR,
        "prefix": "glcm",
        "dimension": 40,
        "dictionary": "glcm_feature_dictionary.json"
    },
    {
        "name": "hog",
        "directory": HANDCRAFTED_DIR,
        "prefix": "hog",
        "dimension": 8100,
        "dictionary": "hog_feature_dictionary.json"
    },
    {
        "name": "pooled_hog",
        "directory": HANDCRAFTED_DIR,
        "prefix": "pooled_hog",
        "dimension": 144,
        "dictionary": "pooled_hog_feature_dictionary.json"
    },
    {
        "name": "transformer",
        "directory": TRANSFORMER_DIR,
        "prefix": "transformer",
        "dimension": 768,
        "dictionary": "transformer_feature_dictionary.json"
    },
    {
        "name": "bovw_200",
        "directory": BOVW_DIR,
        "prefix": "bovw_200",
        "dimension": 200,
        "dictionary": "bovw_200_feature_dictionary.json"
    }
]


EXPECTED_DIMENSION = sum(
    group["dimension"] for group in FEATURE_GROUPS
)


def combine_split(split):

    matrices = []
    reference_labels = None

    print(f"\nCOMBINING {split.upper()}")
    print("=" * 65)

    for group in FEATURE_GROUPS:

        directory = group["directory"]
        prefix = group["prefix"]

        X = np.load(
            directory / f"{prefix}_{split}.npy"
        )

        y = np.load(
            directory / f"{prefix}_{split}_labels.npy"
        )

        if X.ndim != 2:
            raise ValueError(f"{prefix}: expected 2D features")

        if X.shape[1] != group["dimension"]:
            raise ValueError(
                f"{prefix}: incorrect feature dimension "
                f"{X.shape[1]}"
            )

        if len(X) != len(y):
            raise ValueError(
                f"{prefix}: feature-label count mismatch"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"{prefix}: non-finite values found"
            )

        if reference_labels is None:
            reference_labels = y.copy()

        else:
            np.testing.assert_array_equal(
                reference_labels,
                y,
                err_msg=f"Label mismatch in {prefix}"
            )

        matrices.append(X)

        print(
            f"{group['name']:15s}: {X.shape}"
        )

    combined = np.concatenate(
        matrices,
        axis=1
    ).astype(np.float32)

    if combined.shape[1] != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Combined feature dimension mismatch"
        )

    np.save(
        OUTPUT_DIR / f"all_{split}.npy",
        combined
    )

    np.save(
        OUTPUT_DIR / f"all_{split}_labels.npy",
        reference_labels
    )

    print("-" * 65)
    print(f"ALL {split.upper()}: {combined.shape}")

    return combined.shape


def create_feature_dictionary():

    dictionary = []
    column = 0

    for group in FEATURE_GROUPS:

        dictionary_path = (
            group["directory"] / group["dictionary"]
        )

        with open(dictionary_path, "r") as file:
            entries = json.load(file)

        if len(entries) != group["dimension"]:
            raise ValueError(
                f"Incorrect dictionary length: {group['name']}"
            )

        for local_index, entry in enumerate(entries):

            item = dict(entry)

            item["local_index"] = local_index
            item["index"] = column
            item["feature_group"] = group["name"]

            dictionary.append(item)

            column += 1

    assert len(dictionary) == EXPECTED_DIMENSION

    output_path = (
        OUTPUT_DIR / "all_feature_dictionary.json"
    )

    with open(output_path, "w") as file:
        json.dump(dictionary, file, indent=2)

    print(
        f"\nFeature dictionary: {len(dictionary)} columns"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("ALL FEATURE FAMILIES CONCATENATION")
    print("=" * 65)

    train_shape = combine_split("train")
    val_shape = combine_split("val")

    create_feature_dictionary()

    metadata = {
        "train_shape": list(train_shape),
        "val_shape": list(val_shape),
        "total_features": EXPECTED_DIMENSION,
        "feature_groups": [
            {
                "name": group["name"],
                "dimension": group["dimension"]
            }
            for group in FEATURE_GROUPS
        ],
        "selected_bovw_vocabulary": 200,
        "split_selection": "TRAIN and VAL only",
        "test_used_for_selection": False
    }

    with open(
        OUTPUT_DIR / "all_features_metadata.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 65)
    print("ALL FEATURES COMBINED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
