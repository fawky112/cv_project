import json
import time
from pathlib import Path

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "features" / "bovw"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

VOCAB_SIZES = [100, 200, 500]

MAX_DESCRIPTORS_PER_IMAGE = 200

MAX_VOCAB_TRAINING_DESCRIPTORS = 60000

SEED = 42

BATCH_SIZE = 2048


def load_sift_cache(split):

    descriptors = np.load(
        FEATURE_ROOT / f"sift_{split}_descriptors.npy",
        mmap_mode="r"
    )

    offsets = np.load(
        FEATURE_ROOT / f"sift_{split}_offsets.npy"
    )

    names = np.load(
        SPLIT_ROOT / f"{split}_names.npy"
    )

    labels = np.load(
        SPLIT_ROOT / f"{split}_labels.npy"
    )

    if len(offsets) != len(names) + 1:
        raise ValueError(
            f"Offsets and image count mismatch: {split}"
        )

    if offsets[0] != 0 or offsets[-1] != len(descriptors):
        raise ValueError(
            f"Invalid descriptor offsets: {split}"
        )

    return descriptors, offsets, labels


def sample_vocabulary_descriptors(
    descriptors,
    offsets
):

    rng = np.random.default_rng(SEED)

    samples = []

    print("\nSampling TRAIN descriptors for vocabulary...")

    for image_index in range(len(offsets) - 1):

        start = offsets[image_index]
        end = offsets[image_index + 1]

        count = end - start

        if count == 0:
            continue

        sample_count = min(
            count,
            MAX_DESCRIPTORS_PER_IMAGE
        )

        if count > sample_count:

            indices = rng.choice(
                count,
                size=sample_count,
                replace=False
            )

            selected = descriptors[
                start + indices
            ]

        else:

            selected = descriptors[start:end]

        samples.append(
            np.asarray(selected, dtype=np.float32)
        )

    if not samples:
        raise RuntimeError("No TRAIN SIFT descriptors found")

    sampled = np.concatenate(
        samples,
        axis=0
    )

    print(
        f"Descriptors after per-image cap: "
        f"{len(sampled)}"
    )

    if len(sampled) > MAX_VOCAB_TRAINING_DESCRIPTORS:

        indices = rng.choice(
            len(sampled),
            size=MAX_VOCAB_TRAINING_DESCRIPTORS,
            replace=False
        )

        sampled = sampled[indices]

    print(
        f"Final vocabulary training sample: "
        f"{sampled.shape}"
    )

    return sampled


def build_histograms(
    descriptors,
    offsets,
    centers,
    split,
    vocab_size
):

    num_images = len(offsets) - 1

    X = np.zeros(
        (num_images, vocab_size),
        dtype=np.float32
    )

    start_time = time.perf_counter()

    print(
        f"\nBuilding {split.upper()} "
        f"BoVW-{vocab_size} histograms..."
    )

    for image_index in range(num_images):

        start = offsets[image_index]
        end = offsets[image_index + 1]

        image_descriptors = descriptors[start:end]

        if len(image_descriptors) > 0:

            words = pairwise_distances_argmin(
                image_descriptors,
                centers,
                metric="euclidean"
            )

            histogram = np.bincount(
                words,
                minlength=vocab_size
            ).astype(np.float32)

            total = histogram.sum()

            if total > 0:
                histogram /= total

            X[image_index] = histogram

        if (
            (image_index + 1) % 200 == 0
            or image_index + 1 == num_images
        ):

            print(
                f"\r{split.upper()} BoVW-{vocab_size}: "
                f"{image_index + 1}/{num_images}",
                end="",
                flush=True
            )

    print()

    elapsed = time.perf_counter() - start_time

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Non-finite BoVW features")

    row_sums = X.sum(axis=1)

    nonempty = np.diff(offsets) > 0

    np.testing.assert_allclose(
        row_sums[nonempty],
        1.0,
        atol=1e-5
    )

    np.testing.assert_allclose(
        row_sums[~nonempty],
        0.0,
        atol=1e-5
    )

    print(f"Feature Matrix: {X.shape}")

    print(
        f"Histogram Time/Image: "
        f"{elapsed / num_images * 1000:.3f} ms"
    )

    return X, elapsed


def save_dictionary(vocab_size):

    dictionary = []

    for index in range(vocab_size):

        dictionary.append({
            "index": index,
            "name": f"bovw_{vocab_size}_word_{index}",
            "method": "SIFT_BoVW",
            "computation": (
                "L1-normalized count of SIFT descriptors "
                "assigned to nearest visual word"
            ),
            "vocabulary_size": vocab_size,
            "visual_word": index,
            "descriptor_dimension": 128,
            "preprocessing": (
                "Resize 224x224, grayscale SIFT"
            ),
            "vocabulary_training_split": "TRAIN",
            "max_descriptors_per_train_image": (
                MAX_DESCRIPTORS_PER_IMAGE
            ),
            "max_total_vocabulary_training_descriptors": (
                MAX_VOCAB_TRAINING_DESCRIPTORS
            )
        })

    path = (
        FEATURE_ROOT
        / f"bovw_{vocab_size}_feature_dictionary.json"
    )

    with open(path, "w") as file:
        json.dump(dictionary, file, indent=2)


def main():

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BAG OF VISUAL WORDS - VOCABULARY TRAINING")
    print("=" * 70)

    train_desc, train_offsets, train_labels = (
        load_sift_cache("train")
    )

    val_desc, val_offsets, val_labels = (
        load_sift_cache("val")
    )

    training_sample = sample_vocabulary_descriptors(
        train_desc,
        train_offsets
    )

    results = {}

    for vocab_size in VOCAB_SIZES:

        print("\n" + "=" * 70)
        print(f"BUILDING VOCABULARY: K = {vocab_size}")
        print("=" * 70)

        model = MiniBatchKMeans(
            n_clusters=vocab_size,
            random_state=SEED,
            batch_size=BATCH_SIZE,
            n_init=3,
            max_iter=100
        )

        start = time.perf_counter()

        model.fit(training_sample)

        clustering_time = time.perf_counter() - start

        centers = model.cluster_centers_.astype(
            np.float32
        )

        np.save(
            FEATURE_ROOT
            / f"bovw_{vocab_size}_centers.npy",
            centers
        )

        print(
            f"Clustering Time: "
            f"{clustering_time:.2f} sec"
        )

        X_train, train_hist_time = build_histograms(
            train_desc,
            train_offsets,
            centers,
            "train",
            vocab_size
        )

        X_val, val_hist_time = build_histograms(
            val_desc,
            val_offsets,
            centers,
            "val",
            vocab_size
        )

        np.save(
            FEATURE_ROOT / f"bovw_{vocab_size}_train.npy",
            X_train
        )

        np.save(
            FEATURE_ROOT / f"bovw_{vocab_size}_val.npy",
            X_val
        )

        np.save(
            FEATURE_ROOT / f"bovw_{vocab_size}_train_labels.npy",
            train_labels
        )

        np.save(
            FEATURE_ROOT / f"bovw_{vocab_size}_val_labels.npy",
            val_labels
        )

        save_dictionary(vocab_size)

        results[str(vocab_size)] = {
            "vocabulary_size": vocab_size,
            "training_descriptors": len(training_sample),
            "clustering_time_seconds": clustering_time,
            "train_histogram_seconds": train_hist_time,
            "val_histogram_seconds": val_hist_time,
            "train_shape": list(X_train.shape),
            "val_shape": list(X_val.shape)
        }

    with open(
        FEATURE_ROOT / "bovw_experiment_metadata.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("ALL BOVW VOCABULARIES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
