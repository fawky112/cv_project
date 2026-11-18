import json
import time
from pathlib import Path

import cv2
import numpy as np

from sklearn.metrics import pairwise_distances_argmin

from src.features.color import extract_color_features
from src.features.lbp import extract_lbp_features
from src.features.glcm import extract_glcm_features
from src.features.hog import extract_both_hog_features
from src.features.sift import extract_sift_descriptors


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_DIR = ROOT / "data" / "splits"
HANDCRAFTED_DIR = ROOT / "features" / "handcrafted"
BOVW_DIR = ROOT / "features" / "bovw"
COMBINED_DIR = ROOT / "features" / "combined"
TRANSFORMER_DIR = ROOT / "features" / "transformer"

FEATURE_DIMS = {
    "color": 88,
    "lbp": 1003,
    "glcm": 40,
    "hog": 8100,
    "pooled_hog": 144,
    "bovw_200": 200,
}


def load_vocabulary():
    centers = np.load(
        BOVW_DIR / "bovw_200_centers.npy"
    ).astype(np.float32)

    if centers.shape != (200, 128):
        raise ValueError(
            f"Invalid vocabulary shape: {centers.shape}"
        )

    return centers


def make_bovw_histogram(descriptors, centers):

    histogram = np.zeros(200, dtype=np.float32)

    if len(descriptors) == 0:
        return histogram

    words = pairwise_distances_argmin(
        descriptors,
        centers,
        metric="euclidean"
    )

    histogram = np.bincount(
        words,
        minlength=200
    ).astype(np.float32)

    histogram /= histogram.sum()

    return histogram


def extract_test_features():

    names = np.load(
        SPLIT_DIR / "test_names.npy"
    )

    labels = np.load(
        SPLIT_DIR / "test_labels.npy"
    )

    if len(names) != 3669:
        raise ValueError(
            f"Expected 3669 images, found {len(names)}"
        )

    centers = load_vocabulary()

    HANDCRAFTED_DIR.mkdir(parents=True, exist_ok=True)
    BOVW_DIR.mkdir(parents=True, exist_ok=True)

    matrices = {
        name: np.zeros(
            (len(names), dimension),
            dtype=np.float32
        )
        for name, dimension in FEATURE_DIMS.items()
    }

    times = {
        name: []
        for name in FEATURE_DIMS
    }

    print("=" * 70)
    print("EXTRACTING HANDCRAFTED + BOVW TEST FEATURES")
    print("=" * 70)

    for index, name in enumerate(names):

        image_path = (
            DATA_ROOT / "images" / f"{name}.jpg"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        image_128 = cv2.resize(
            image,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )

        image_224 = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        start = time.perf_counter()
        matrices["color"][index] = extract_color_features(
            image_128
        )
        times["color"].append(time.perf_counter() - start)

        start = time.perf_counter()
        matrices["lbp"][index] = extract_lbp_features(
            image_128,
            radius=1,
            neighbours=8,
            grid=(4, 4)
        )
        times["lbp"].append(time.perf_counter() - start)

        start = time.perf_counter()
        matrices["glcm"][index] = extract_glcm_features(
            image_128
        )
        times["glcm"].append(time.perf_counter() - start)

        start = time.perf_counter()

        hog, pooled_hog = extract_both_hog_features(
            image_128
        )

        hog_elapsed = time.perf_counter() - start

        matrices["hog"][index] = hog
        matrices["pooled_hog"][index] = pooled_hog

        # These descriptors share gradient computation.
        times["hog"].append(hog_elapsed)
        times["pooled_hog"].append(hog_elapsed)

        start = time.perf_counter()

        _, descriptors = extract_sift_descriptors(
            image_224
        )

        matrices["bovw_200"][index] = make_bovw_histogram(
            descriptors,
            centers
        )

        times["bovw_200"].append(
            time.perf_counter() - start
        )

        if (index + 1) % 200 == 0 or index + 1 == len(names):

            print(
                f"\rTEST: {index + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    timing_summary = {}

    for name, matrix in matrices.items():

        assert matrix.shape == (
            len(names),
            FEATURE_DIMS[name]
        )

        assert np.all(np.isfinite(matrix))

        output_dir = (
            BOVW_DIR
            if name == "bovw_200"
            else HANDCRAFTED_DIR
        )

        np.save(
            output_dir / f"{name}_test.npy",
            matrix
        )

        np.save(
            output_dir / f"{name}_test_labels.npy",
            labels
        )

        average_ms = float(
            np.mean(times[name]) * 1000
        )

        timing_summary[name] = average_ms

        print(
            f"{name:12s} | "
            f"{str(matrix.shape):18s} | "
            f"{average_ms:.3f} ms/image"
        )

    handcrafted = np.concatenate(
        [
            matrices["color"],
            matrices["lbp"],
            matrices["glcm"],
            matrices["hog"],
            matrices["pooled_hog"]
        ],
        axis=1
    ).astype(np.float32)

    assert handcrafted.shape == (3669, 9375)

    np.save(
        HANDCRAFTED_DIR / "handcrafted_test.npy",
        handcrafted
    )

    np.save(
        HANDCRAFTED_DIR / "handcrafted_test_labels.npy",
        labels
    )

    # Build the complete feature vector in the same order
    # used during TRAIN and VAL concatenation.

    transformer = np.load(
        TRANSFORMER_DIR / "transformer_test.npy"
    )

    transformer_labels = np.load(
        TRANSFORMER_DIR / "transformer_test_labels.npy"
    )

    np.testing.assert_array_equal(
        labels,
        transformer_labels
    )

    assert transformer.shape == (3669, 768)

    combined = np.concatenate(
        [
            handcrafted,
            transformer,
            matrices["bovw_200"]
        ],
        axis=1
    ).astype(np.float32)

    assert combined.shape == (3669, 10343)

    COMBINED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        COMBINED_DIR / "all_test.npy",
        combined
    )

    np.save(
        COMBINED_DIR / "all_test_labels.npy",
        labels
    )

    print("\nCOMBINED MATRICES")
    print("-" * 70)

    print(f"Handcrafted TEST: {handcrafted.shape}")
    print(f"All Features TEST: {combined.shape}")

    with open(
        ROOT / "results" / "tables"
        / "remaining_test_extraction_times.json",
        "w"
    ) as file:

        json.dump(
            {
                "ms_per_image": timing_summary,
                "timing_note": (
                    "Image loading and resizing excluded. "
                    "HOG and pooled HOG times are shared, "
                    "not independent. BoVW time includes SIFT "
                    "and visual-word assignment."
                )
            },
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("REMAINING TEST FEATURE EXTRACTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    extract_test_features()
