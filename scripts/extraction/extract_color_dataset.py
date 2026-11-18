import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.features.color import (
    extract_color_features,
    get_color_feature_names
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "handcrafted"

IMAGE_SIZE = 128


def extract_split(split_name):

    names = np.load(
        SPLIT_ROOT / f"{split_name}_names.npy"
    )

    labels = np.load(
        SPLIT_ROOT / f"{split_name}_labels.npy"
    )

    features = np.zeros(
        (len(names), 88),
        dtype=np.float32
    )

    extraction_times = []

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

        feature_vector = extract_color_features(image)

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

    assert features.shape == (len(names), 88)

    assert np.all(np.isfinite(features))

    np.save(
        OUTPUT_ROOT / f"color_{split_name}.npy",
        features
    )

    np.save(
        OUTPUT_ROOT / f"color_{split_name}_labels.npy",
        labels
    )

    avg_time = np.mean(extraction_times) * 1000

    print(f"Feature Matrix: {features.shape}")

    print(f"Average Extraction Time: {avg_time:.3f} ms/image")

    return avg_time


def save_feature_dictionary():

    names = get_color_feature_names()

    dictionary = []

    for index, name in enumerate(names):

        if "hist" in name:
            computation = "L1-normalized histogram bin"

        else:
            computation = name.split("_")[-1]

        dictionary.append({
            "index": index,
            "name": name,
            "method": "handcrafted_color",
            "computation": computation,
            "preprocessing": "Resize to 128x128; BGR to RGB/HSV",
            "resolution": [128, 128]
        })

    output_path = OUTPUT_ROOT / "color_feature_dictionary.json"

    with open(output_path, "w") as file:
        json.dump(dictionary, file, indent=4)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for split in ["train", "val", "test"]:

        print(f"\nExtracting {split.upper()} Features...")

        avg_time = extract_split(split)

        results[split] = avg_time

    save_feature_dictionary()

    with open(
        OUTPUT_ROOT / "color_extraction_times.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    print("\nCOLOR FEATURE EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
