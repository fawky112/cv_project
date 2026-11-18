import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.features.lbp import extract_lbp_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "handcrafted"

IMAGE_SIZE = 128

RADIUS = 1
NEIGHBOURS = 8
GRID = (4, 4)

FEATURE_COUNT = 59 * (1 + GRID[0] * GRID[1])


def extract_split(split_name):

    names = np.load(
        SPLIT_ROOT / f"{split_name}_names.npy"
    )

    labels = np.load(
        SPLIT_ROOT / f"{split_name}_labels.npy"
    )

    features = np.zeros(
        (len(names), FEATURE_COUNT),
        dtype=np.float32
    )

    extraction_times = []

    print(f"\nExtracting {split_name.upper()} LBP Features...")

    for i, image_name in enumerate(names):

        image_path = DATA_ROOT / "images" / f"{image_name}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        start = time.perf_counter()

        feature_vector = extract_lbp_features(
            image,
            radius=RADIUS,
            neighbours=NEIGHBOURS,
            grid=GRID
        )

        elapsed = time.perf_counter() - start

        features[i] = feature_vector

        extraction_times.append(elapsed)

        if (i + 1) % 200 == 0 or i + 1 == len(names):

            print(
                f"\r{split_name.upper()}: "
                f"{i + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    assert features.shape == (
        len(names),
        FEATURE_COUNT
    )

    assert np.all(np.isfinite(features))

    np.save(
        OUTPUT_ROOT / f"lbp_{split_name}.npy",
        features
    )

    np.save(
        OUTPUT_ROOT / f"lbp_{split_name}_labels.npy",
        labels
    )

    avg_time = float(np.mean(extraction_times) * 1000)

    print(f"Feature Matrix: {features.shape}")

    print(
        f"Average Extraction Time: "
        f"{avg_time:.3f} ms/image"
    )

    return avg_time


def save_feature_dictionary():

    dictionary = []

    index = 0

    def add_histogram(location, row=None, col=None):

        nonlocal index

        for bin_idx in range(59):

            name = f"lbp_{location}_bin_{bin_idx}"

            dictionary.append({
                "index": index,
                "name": name,
                "method": "handcrafted_lbp",
                "computation": (
                    "L1-normalized uniform LBP histogram"
                ),
                "location": location,
                "cell_row": row,
                "cell_col": col,
                "bin": bin_idx,
                "radius": RADIUS,
                "neighbours": NEIGHBOURS,
                "preprocessing": (
                    "Resize 128x128, convert BGR to grayscale"
                ),
                "resolution": [128, 128]
            })

            index += 1

    add_histogram("full")

    for row in range(GRID[0]):

        for col in range(GRID[1]):

            add_histogram(
                "cell",
                row=row,
                col=col
            )

    assert len(dictionary) == FEATURE_COUNT

    output_path = (
        OUTPUT_ROOT / "lbp_feature_dictionary.json"
    )

    with open(output_path, "w") as file:

        json.dump(
            dictionary,
            file,
            indent=4
        )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    # Extract TRAIN and VAL for model selection.
    # TEST will be extracted after settings are finalized.

    for split in ["train", "val"]:

        avg_time = extract_split(split)

        results[split] = avg_time

    save_feature_dictionary()

    with open(
        OUTPUT_ROOT / "lbp_extraction_times.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("LBP FEATURE EXTRACTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
