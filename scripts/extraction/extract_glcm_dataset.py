import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.features.glcm import (
    extract_glcm_features,
    get_glcm_feature_names,
    GRAY_LEVELS,
    DISTANCES,
    ANGLES
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "handcrafted"

IMAGE_SIZE = 128
FEATURE_COUNT = 40


def extract_split(split):

    names = np.load(
        SPLIT_ROOT / f"{split}_names.npy"
    )

    labels = np.load(
        SPLIT_ROOT / f"{split}_labels.npy"
    )

    X = np.zeros(
        (len(names), FEATURE_COUNT),
        dtype=np.float32
    )

    times = []

    print(f"\nExtracting {split.upper()} GLCM Features...")

    for i, name in enumerate(names):

        path = DATA_ROOT / "images" / f"{name}.jpg"

        image = cv2.imread(str(path))

        if image is None:
            raise FileNotFoundError(path)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        start = time.perf_counter()

        X[i] = extract_glcm_features(image)

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        if (i + 1) % 200 == 0 or i + 1 == len(names):

            print(
                f"\r{split.upper()}: {i + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    assert X.shape == (len(names), FEATURE_COUNT)
    assert np.all(np.isfinite(X))

    np.save(
        OUTPUT_ROOT / f"glcm_{split}.npy",
        X
    )

    np.save(
        OUTPUT_ROOT / f"glcm_{split}_labels.npy",
        labels
    )

    avg_time = float(np.mean(times) * 1000)

    print(f"Feature Matrix: {X.shape}")
    print(f"Average Extraction Time: {avg_time:.3f} ms/image")

    return avg_time


def save_dictionary():

    names = get_glcm_feature_names()

    dictionary = []

    for index, name in enumerate(names):

        parts = name.split("_")

        distance = int(parts[1][1:])
        angle = int(parts[2][1:])
        statistic = parts[3]

        dictionary.append({
            "index": index,
            "name": name,
            "method": "handcrafted_glcm",
            "computation": statistic,
            "distance": distance,
            "angle_degrees": angle,
            "gray_levels": GRAY_LEVELS,
            "symmetric": True,
            "normalized": True,
            "preprocessing": (
                "Resize 128x128, grayscale, "
                "quantize to 32 gray levels"
            ),
            "resolution": [128, 128]
        })

    with open(
        OUTPUT_ROOT / "glcm_feature_dictionary.json",
        "w"
    ) as file:

        json.dump(dictionary, file, indent=4)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    timing = {}

    for split in ["train", "val"]:

        timing[split] = extract_split(split)

    save_dictionary()

    with open(
        OUTPUT_ROOT / "glcm_extraction_times.json",
        "w"
    ) as file:

        json.dump(timing, file, indent=4)

    print("\n" + "=" * 60)
    print("GLCM FEATURE EXTRACTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
