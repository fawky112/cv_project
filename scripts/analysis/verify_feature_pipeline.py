from pathlib import Path

import cv2
import numpy as np

from src.features.color import extract_color_features
from src.features.lbp import extract_lbp_features
from src.features.glcm import extract_glcm_features
from src.features.hog import extract_both_hog_features


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"
SPLIT_DIR = ROOT / "data" / "splits"
FEATURE_DIR = ROOT / "features" / "handcrafted"

SEED = 42
SAMPLES_PER_SPLIT = 20


def load_official_labels():

    annotations = {}

    path = DATA_ROOT / "annotations" / "list.txt"

    with open(path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            name = parts[0]
            label = int(parts[1]) - 1

            annotations[name] = label

    return annotations


def extract_features(image):

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    hog, pooled = extract_both_hog_features(image)

    return {
        "color": extract_color_features(image),
        "lbp": extract_lbp_features(
            image,
            radius=1,
            neighbours=8,
            grid=(4, 4)
        ),
        "glcm": extract_glcm_features(image),
        "hog": hog,
        "pooled_hog": pooled
    }


def verify_split(split, official_labels):

    print("\n" + "=" * 75)
    print(f"VERIFYING {split.upper()}")
    print("=" * 75)

    names = np.load(
        SPLIT_DIR / f"{split}_names.npy"
    )

    labels = np.load(
        SPLIT_DIR / f"{split}_labels.npy"
    )

    feature_sets = [
        "color",
        "lbp",
        "glcm",
        "hog",
        "pooled_hog"
    ]

    stored = {
        feature_set: np.load(
            FEATURE_DIR / f"{feature_set}_{split}.npy",
            mmap_mode="r"
        )
        for feature_set in feature_sets
    }

    print(f"Images: {len(names)}")

    label_errors = []

    for i, name in enumerate(names):

        name = str(name)

        if name not in official_labels:
            label_errors.append((name, "missing annotation"))
            continue

        if int(labels[i]) != official_labels[name]:

            label_errors.append((
                name,
                int(labels[i]),
                official_labels[name]
            ))

    print(f"Annotation label mismatches: {len(label_errors)}")

    if label_errors:
        print("Examples:", label_errors[:5])

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(names),
        size=min(SAMPLES_PER_SPLIT, len(names)),
        replace=False
    )

    mismatches = {name: [] for name in feature_sets}

    for i in indices:

        name = str(names[i])

        path = DATA_ROOT / "images" / f"{name}.jpg"

        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(path)

        extracted = extract_features(image)

        for feature_set in feature_sets:

            original = stored[feature_set][i]

            recalculated = extracted[feature_set]

            if not np.allclose(
                original,
                recalculated,
                rtol=1e-4,
                atol=1e-5
            ):

                max_difference = float(
                    np.max(
                        np.abs(original - recalculated)
                    )
                )

                mismatches[feature_set].append(
                    (name, max_difference)
                )

    print(f"\nFeature samples checked: {len(indices)}")

    for feature_set in feature_sets:

        errors = mismatches[feature_set]

        status = "PASS" if not errors else "MISMATCH"

        print(
            f"{feature_set:12s} | "
            f"{status:8s} | "
            f"Different samples: {len(errors)}"
        )

        for name, difference in errors[:3]:

            print(
                f"    {name}: "
                f"max absolute difference = {difference:.6f}"
            )

    return len(label_errors), mismatches


def main():

    annotations = load_official_labels()

    all_passed = True

    for split in ["train", "val", "test"]:

        label_errors, mismatches = verify_split(
            split,
            annotations
        )

        if label_errors:
            all_passed = False

        if any(mismatches.values()):
            all_passed = False

    print("\n" + "=" * 75)

    if all_passed:
        print("PIPELINE INTEGRITY CHECK PASSED")
    else:
        print("PIPELINE INTEGRITY CHECK FOUND DIFFERENCES")
        print("Review mismatches before finalizing the report.")

    print("=" * 75)


if __name__ == "__main__":
    main()
