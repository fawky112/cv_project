import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from src.classifiers.knn import (
    StandardScalerScratch,
    KNNFromScratch
)


ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


NUM_CLASSES = 37
EXPECTED_TRAIN = 2944
EXPECTED_TEST = 3669


CONFIGURATIONS = [
    {
        "name": "transformer",
        "directory": ROOT / "features" / "transformer",
        "k": 5,
        "distance": "cosine"
    },
    {
        "name": "mrmr_20",
        "directory": ROOT / "features" / "selected",
        "k": 5,
        "distance": "cosine"
    },
    {
        "name": "mrmr_50",
        "directory": ROOT / "features" / "selected",
        "k": 5,
        "distance": "cosine"
    },
    {
        "name": "mrmr_100",
        "directory": ROOT / "features" / "selected",
        "k": 3,
        "distance": "cosine"
    }
]


def calculate_metrics(y_true, y_pred):

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = np.sum(
            (y_true == class_id)
            & (y_pred == class_id)
        )

        fp = np.sum(
            (y_true != class_id)
            & (y_pred == class_id)
        )

        fn = np.sum(
            (y_true == class_id)
            & (y_pred != class_id)
        )

        denominator = 2 * tp + fp + fn

        if denominator > 0:
            f1 = 2 * tp / denominator
        else:
            f1 = 0.0

        f1_scores.append(f1)

    macro_f1 = float(
        np.mean(f1_scores)
    )

    return accuracy, macro_f1


def save_confusion_matrix(name, y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    np.save(
        RESULT_DIR / f"{name}_test_confusion_matrix.npy",
        cm
    )

    np.savetxt(
        RESULT_DIR / f"{name}_test_confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=","
    )

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=np.float64
        ),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    im = ax.imshow(
        normalized,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"{name.upper()} - TEST Confusion Matrix",
        fontsize=15
    )

    ax.set_xlabel(
        "Predicted Breed ID",
        fontsize=12
    )

    ax.set_ylabel(
        "True Breed ID",
        fontsize=12
    )

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_yticks(
        np.arange(NUM_CLASSES)
    )

    ax.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=6
    )

    ax.tick_params(
        axis="y",
        labelsize=6
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{name}_test_confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return cm


def load_features(config):

    name = config["name"]
    directory = config["directory"]

    X_train = np.load(
        directory / f"{name}_train.npy"
    )

    y_train = np.load(
        directory / f"{name}_train_labels.npy"
    )

    X_test = np.load(
        directory / f"{name}_test.npy"
    )

    y_test = np.load(
        directory / f"{name}_test_labels.npy"
    )

    if len(X_train) != EXPECTED_TRAIN:
        raise ValueError(
            f"{name}: incorrect TRAIN size"
        )

    if len(X_test) != EXPECTED_TEST:
        raise ValueError(
            f"{name}: incorrect TEST size"
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            f"{name}: TRAIN labels mismatch"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{name}: TEST labels mismatch"
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{name}: feature dimension mismatch"
        )

    if not np.all(np.isfinite(X_train)):
        raise ValueError(
            f"{name}: invalid TRAIN features"
        )

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            f"{name}: invalid TEST features"
        )

    return X_train, y_train, X_test, y_test


def evaluate(config):

    name = config["name"]

    print("\n" + "=" * 75)
    print(f"FINAL TEST: {name.upper()}")
    print("=" * 75)

    X_train, y_train, X_test, y_test = load_features(
        config
    )

    print(f"TRAIN Matrix: {X_train.shape}")
    print(f"TEST Matrix:  {X_test.shape}")

    print("\nStandardizing features using TRAIN statistics...")

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    print("Initializing K-NN classifier...")

    model = KNNFromScratch(
        k=config["k"],
        distance=config["distance"]
    )

    model.fit(
        X_train,
        y_train
    )

    print("Predicting TEST images...")

    start = time.perf_counter()

    predictions = model.predict(
        X_test,
        batch_size=64
    )

    elapsed = time.perf_counter() - start

    accuracy, macro_f1 = calculate_metrics(
        y_test,
        predictions
    )

    cm = save_confusion_matrix(
        name,
        y_test,
        predictions
    )

    np.save(
        RESULT_DIR / f"{name}_test_predictions.npy",
        predictions
    )

    correct_predictions = int(
        np.sum(y_test == predictions)
    )

    result = {
        "model": name,
        "features": int(X_train.shape[1]),
        "k": config["k"],
        "distance": config["distance"],
        "test_images": int(len(y_test)),
        "correct_predictions": correct_predictions,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "prediction_ms_per_image": (
            elapsed / len(y_test) * 1000
        )
    }

    print("\nRESULTS")
    print("-" * 75)

    print(f"Features: {result['features']}")
    print(f"K: {result['k']}")
    print(f"Distance: {result['distance']}")

    print(
        f"Correct Predictions: "
        f"{correct_predictions}/{len(y_test)}"
    )

    print(
        f"TEST Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"TEST Macro-F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Prediction Time: "
        f"{result['prediction_ms_per_image']:.3f} ms/image"
    )

    print(
        f"Confusion Matrix Shape: "
        f"{cm.shape}"
    )

    print(
        f"Results saved for: {name}"
    )

    return result


def save_results(results):

    csv_path = (
        RESULT_DIR / "final_test_results.csv"
    )

    json_path = (
        RESULT_DIR / "final_test_results.json"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    return csv_path, json_path


def print_summary(results):

    print("\n" + "=" * 100)
    print("FINAL TEST RESULTS - OXFORD-IIIT PET")
    print("=" * 100)

    print(
        f"{'Model':<18}"
        f"{'Features':>10}"
        f"{'K':>6}"
        f"{'Distance':>12}"
        f"{'Accuracy':>14}"
        f"{'Macro-F1':>12}"
        f"{'Time(ms)':>12}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['model']:<18}"
            f"{result['features']:>10}"
            f"{result['k']:>6}"
            f"{result['distance']:>12}"
            f"{result['test_accuracy'] * 100:>13.2f}%"
            f"{result['test_macro_f1']:>12.4f}"
            f"{result['prediction_ms_per_image']:>12.3f}"
        )

    print("=" * 100)


def main():

    print("=" * 75)
    print("CSE480 MACHINE VISION")
    print("MILESTONE 1 - FINAL TEST EVALUATION")
    print("=" * 75)

    results = []

    for config in CONFIGURATIONS:

        result = evaluate(
            config
        )

        results.append(
            result
        )

    csv_path, json_path = save_results(
        results
    )

    print_summary(
        results
    )

    print("\nResults saved to:")
    print(csv_path)
    print(json_path)

    print("\nConfusion matrices saved to:")
    print(FIGURE_DIR)

    print("\n" + "=" * 75)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
