import csv
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from preprocessing import (
    mean_filter,
    gaussian_filter,
    median_filter,
    histogram_equalization,
    contrast_stretching,
    intensity_normalization,
    resize_image
)

from src.features.color import extract_color_features

from src.classifiers.knn import (
    StandardScalerScratch,
    KNNFromScratch
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

FEATURE_ROOT = PROJECT_ROOT / "features" / "preprocessing"

RESULTS_ROOT = PROJECT_ROOT / "results"

TABLE_ROOT = RESULTS_ROOT / "tables"
FIGURE_ROOT = RESULTS_ROOT / "figures"

IMAGE_SIZE = 128
FEATURE_COUNT = 88

EXPERIMENTS = [
    "baseline",
    "mean",
    "gaussian",
    "median",
    "histogram_equalization",
    "contrast_stretching",
    "intensity_normalization",
    "resize_96",
    "resize_64"
]


def preprocess_image(image, method):

    if method == "resize_96":
        return resize_image(image, 96)

    if method == "resize_64":
        return resize_image(image, 64)

    image = resize_image(image, IMAGE_SIZE)

    if method == "baseline":
        return image

    # Apply processing to luminance only.
    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )

    y = ycrcb[:, :, 0]

    if method == "mean":

        processed_y = mean_filter(
            y,
            kernel_size=3
        )

    elif method == "gaussian":

        processed_y = gaussian_filter(
            y,
            kernel_size=3,
            sigma=1.0
        )

    elif method == "median":

        processed_y = median_filter(
            y,
            kernel_size=3
        )

    elif method == "histogram_equalization":

        processed_y = histogram_equalization(y)

    elif method == "contrast_stretching":

        processed_y = contrast_stretching(y)

    elif method == "intensity_normalization":

        z = intensity_normalization(y)

        # Fixed mapping from z-score to uint8.
        processed_y = np.clip(
            127.5 + 40.0 * z,
            0,
            255
        ).astype(np.uint8)

    else:
        raise ValueError(
            f"Unknown preprocessing method: {method}"
        )

    result_ycrcb = ycrcb.copy()

    result_ycrcb[:, :, 0] = processed_y

    result = cv2.cvtColor(
        result_ycrcb,
        cv2.COLOR_YCrCb2BGR
    )

    return result


def extract_features(split_name, method):

    names = np.load(
        SPLIT_ROOT / f"{split_name}_names.npy"
    )

    features = np.zeros(
        (len(names), FEATURE_COUNT),
        dtype=np.float32
    )

    times = []

    for i, image_name in enumerate(names):

        image_path = (
            DATA_ROOT / "images" / f"{image_name}.jpg"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        start = time.perf_counter()

        processed = preprocess_image(
            image,
            method
        )

        vector = extract_color_features(processed)

        elapsed = time.perf_counter() - start

        features[i] = vector

        times.append(elapsed)

        if (i + 1) % 400 == 0 or i + 1 == len(names):

            print(
                f"\r{method:25s} | "
                f"{split_name.upper():5s} | "
                f"{i + 1}/{len(names)}",
                end="",
                flush=True
            )

    print()

    assert np.all(np.isfinite(features))

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        FEATURE_ROOT / f"{method}_{split_name}.npy",
        features
    )

    avg_time = np.mean(times) * 1000

    return features, avg_time


def accuracy_score(y_true, y_pred):

    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true, y_pred):

    scores = []

    for class_id in range(37):

        tp = np.sum(
            (y_true == class_id) &
            (y_pred == class_id)
        )

        fp = np.sum(
            (y_true != class_id) &
            (y_pred == class_id)
        )

        fn = np.sum(
            (y_true == class_id) &
            (y_pred != class_id)
        )

        denominator = 2 * tp + fp + fn

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        scores.append(f1)

    return float(np.mean(scores))


def evaluate_experiment(method, y_train, y_val):

    X_train, train_time = extract_features(
        "train",
        method
    )

    X_val, val_time = extract_features(
        "val",
        method
    )

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    classifier = KNNFromScratch(
        k=1,
        distance="euclidean"
    )

    classifier.fit(X_train, y_train)

    start = time.perf_counter()

    predictions = classifier.predict(X_val)

    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = macro_f1_score(
        y_val,
        predictions
    )

    result = {
        "experiment": method,
        "resolution": (
            96 if method == "resize_96"
            else 64 if method == "resize_64"
            else 128
        ),
        "feature_count": FEATURE_COUNT,
        "val_accuracy": accuracy,
        "val_accuracy_percent": accuracy * 100,
        "val_macro_f1": macro_f1,
        "train_time_ms": train_time,
        "val_time_ms": val_time,
        "prediction_time_ms": (
            prediction_time / len(y_val) * 1000
        )
    }

    return result


def save_results(results):

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        TABLE_ROOT / "preprocessing_results.csv"
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


def plot_results(results):

    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    names = [
        r["experiment"]
        for r in results
    ]

    accuracies = [
        r["val_accuracy_percent"]
        for r in results
    ]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        names,
        accuracies
    )

    plt.axhline(
        accuracies[0],
        linestyle="--",
        label="Baseline 1-NN"
    )

    plt.xlabel("Preprocessing Method")
    plt.ylabel("Validation Accuracy (%)")

    plt.title(
        "Effect of Preprocessing on Color Features + 1-NN"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.legend()

    for bar, value in zip(bars, accuracies):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylim(
        0,
        max(accuracies) * 1.20 + 1
    )

    plt.tight_layout()

    output_path = (
        FIGURE_ROOT / "preprocessing_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(f"Figure saved to: {output_path}")


def main():

    y_train = np.load(
        SPLIT_ROOT / "train_labels.npy"
    )

    y_val = np.load(
        SPLIT_ROOT / "val_labels.npy"
    )

    results = []

    print("=" * 75)
    print("PREPROCESSING EXPERIMENTS")
    print("Color Features + 1-NN")
    print("=" * 75)

    for method in EXPERIMENTS:

        print(f"\nRunning: {method}")

        result = evaluate_experiment(
            method,
            y_train,
            y_val
        )

        results.append(result)

        print(
            f"VAL Accuracy: "
            f"{result['val_accuracy_percent']:.2f}%"
        )

        print(
            f"VAL Macro-F1: "
            f"{result['val_macro_f1']:.4f}"
        )

        print(
            f"TRAIN Extraction: "
            f"{result['train_time_ms']:.3f} ms/image"
        )

    save_results(results)

    plot_results(results)

    best = max(
        results,
        key=lambda r: r["val_accuracy"]
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    for result in results:

        print(
            f"{result['experiment']:25s} | "
            f"Accuracy: "
            f"{result['val_accuracy_percent']:6.2f}% | "
            f"F1: "
            f"{result['val_macro_f1']:.4f}"
        )

    print("\nBEST PREPROCESSING CONFIGURATION")

    print(f"Method: {best['experiment']}")

    print(
        f"VAL Accuracy: "
        f"{best['val_accuracy_percent']:.2f}%"
    )

    print(
        f"VAL Macro-F1: "
        f"{best['val_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
