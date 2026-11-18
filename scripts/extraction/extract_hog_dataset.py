import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.features.hog import extract_both_hog_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "handcrafted"

IMAGE_SIZE = 128

HOG_DIM = 8100
POOLED_DIM = 144


def extract_split(split):

    names = np.load(
        SPLIT_ROOT / f"{split}_names.npy"
    )

    labels = np.load(
        SPLIT_ROOT / f"{split}_labels.npy"
    )

    X_hog = np.zeros(
        (len(names), HOG_DIM),
        dtype=np.float32
    )

    X_pooled = np.zeros(
        (len(names), POOLED_DIM),
        dtype=np.float32
    )

    times = []

    print(f"\nExtracting {split.upper()} HOG Features...")

    for i, name in enumerate(names):

        image_path = DATA_ROOT / "images" / f"{name}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        start = time.perf_counter()

        hog, pooled = extract_both_hog_features(
            image
        )

        elapsed = time.perf_counter() - start

        X_hog[i] = hog
        X_pooled[i] = pooled

        times.append(elapsed)

        if (i + 1) % 200 == 0 or i + 1 == len(names):

            print(
                f"\r{split.upper()}: "
                f"{i + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    assert X_hog.shape == (len(names), HOG_DIM)
    assert X_pooled.shape == (len(names), POOLED_DIM)

    assert np.all(np.isfinite(X_hog))
    assert np.all(np.isfinite(X_pooled))

    np.save(
        OUTPUT_ROOT / f"hog_{split}.npy",
        X_hog
    )

    np.save(
        OUTPUT_ROOT / f"pooled_hog_{split}.npy",
        X_pooled
    )

    np.save(
        OUTPUT_ROOT / f"hog_{split}_labels.npy",
        labels
    )

    np.save(
        OUTPUT_ROOT / f"pooled_hog_{split}_labels.npy",
        labels
    )

    avg_time = float(np.mean(times) * 1000)

    print(f"HOG Matrix: {X_hog.shape}")
    print(f"Pooled HOG Matrix: {X_pooled.shape}")

    print(
        f"Combined Extraction Time: "
        f"{avg_time:.3f} ms/image"
    )

    return avg_time


def save_dictionaries():

    hog_dictionary = []

    index = 0

    for block_y in range(15):
        for block_x in range(15):
            for cell_y in range(2):
                for cell_x in range(2):
                    for bin_idx in range(9):

                        hog_dictionary.append({
                            "index": index,
                            "name": (
                                f"hog_by{block_y}_bx{block_x}"
                                f"_cy{cell_y}_cx{cell_x}"
                                f"_bin{bin_idx}"
                            ),
                            "method": "handcrafted_hog",
                            "computation": (
                                "L2-Hys normalized gradient "
                                "orientation histogram bin"
                            ),
                            "preprocessing": (
                                "Resize 128x128, grayscale"
                            ),
                            "resolution": [128, 128],
                            "cell_size": 8,
                            "block_size": 2,
                            "block_stride": 1,
                            "orientation_bins": 9
                        })

                        index += 1

    pooled_dictionary = []

    index = 0

    for row in range(4):
        for col in range(4):
            for bin_idx in range(9):

                pooled_dictionary.append({
                    "index": index,
                    "name": (
                        f"pooled_hog_r{row}_c{col}"
                        f"_bin{bin_idx}"
                    ),
                    "method": "handcrafted_pooled_hog",
                    "computation": (
                        "Spatial average pooling of "
                        "cell orientation histograms, "
                        "followed by L2 normalization"
                    ),
                    "preprocessing": (
                        "Resize 128x128, grayscale"
                    ),
                    "resolution": [128, 128],
                    "pool_grid": [4, 4],
                    "orientation_bins": 9
                })

                index += 1

    assert len(hog_dictionary) == HOG_DIM
    assert len(pooled_dictionary) == POOLED_DIM

    with open(
        OUTPUT_ROOT / "hog_feature_dictionary.json",
        "w"
    ) as file:
        json.dump(hog_dictionary, file, indent=2)

    with open(
        OUTPUT_ROOT / "pooled_hog_feature_dictionary.json",
        "w"
    ) as file:
        json.dump(pooled_dictionary, file, indent=2)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    timings = {}

    for split in ["train", "val"]:

        timings[split] = extract_split(split)

    save_dictionaries()

    with open(
        OUTPUT_ROOT / "hog_extraction_times.json",
        "w"
    ) as file:

        json.dump(
            {
                "combined_hog_and_pooled_ms_per_image": timings,
                "note": (
                    "Time includes gradient computation, "
                    "cell histograms, HOG blocks and pooling. "
                    "Image reading and resizing are excluded."
                )
            },
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("HOG FEATURE EXTRACTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
