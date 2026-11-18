import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.features.sift import extract_sift_descriptors


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "bovw"

IMAGE_SIZE = 224


def extract_split(split):

    names = np.load(
        SPLIT_ROOT / f"{split}_names.npy"
    )

    all_descriptors = []
    offsets = [0]
    times = []
    keypoint_counts = []

    print("\n" + "=" * 65)
    print(f"EXTRACTING {split.upper()} SIFT DESCRIPTORS")
    print("=" * 65)

    for index, name in enumerate(names):

        image_path = (
            DATA_ROOT / "images" / f"{name}.jpg"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        start = time.perf_counter()

        keypoints, descriptors = extract_sift_descriptors(
            image
        )

        elapsed = time.perf_counter() - start

        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise RuntimeError(
                f"Invalid SIFT shape: {descriptors.shape}"
            )

        if not np.all(np.isfinite(descriptors)):
            raise RuntimeError(
                f"Non-finite SIFT descriptors: {name}"
            )

        all_descriptors.append(descriptors)

        offsets.append(
            offsets[-1] + len(descriptors)
        )

        keypoint_counts.append(len(keypoints))
        times.append(elapsed)

        if (index + 1) % 200 == 0 or index + 1 == len(names):

            print(
                f"\r{split.upper()}: "
                f"{index + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    if offsets[-1] > 0:

        flat_descriptors = np.concatenate(
            all_descriptors,
            axis=0
        ).astype(np.float32)

    else:

        flat_descriptors = np.empty(
            (0, 128),
            dtype=np.float32
        )

    offsets = np.asarray(
        offsets,
        dtype=np.int64
    )

    np.save(
        OUTPUT_ROOT / f"sift_{split}_descriptors.npy",
        flat_descriptors
    )

    np.save(
        OUTPUT_ROOT / f"sift_{split}_offsets.npy",
        offsets
    )

    assert len(offsets) == len(names) + 1
    assert offsets[-1] == len(flat_descriptors)

    avg_time = float(np.mean(times) * 1000)

    print(f"Images: {len(names)}")
    print(f"Total Descriptors: {len(flat_descriptors)}")

    print(
        f"Average Keypoints/Image: "
        f"{np.mean(keypoint_counts):.2f}"
    )

    print(
        f"Images Without Descriptors: "
        f"{np.sum(np.asarray(keypoint_counts) == 0)}"
    )

    print(
        f"Average Extraction Time: "
        f"{avg_time:.3f} ms/image"
    )

    return {
        "images": int(len(names)),
        "total_descriptors": int(len(flat_descriptors)),
        "average_keypoints": float(np.mean(keypoint_counts)),
        "images_without_descriptors": int(
            np.sum(np.asarray(keypoint_counts) == 0)
        ),
        "average_ms_per_image": avg_time
    }


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for split in ["train", "val"]:

        results[split] = extract_split(split)

    with open(
        OUTPUT_ROOT / "sift_extraction_times.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("SIFT DATASET EXTRACTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
