import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.classifiers.knn import (
    StandardScalerScratch,
    KNNFromScratch
)


ROOT = Path(__file__).resolve().parents[2]

HANDCRAFTED_DIR = ROOT / "features" / "handcrafted"
BOVW_DIR = ROOT / "features" / "bovw"
COMBINED_DIR = ROOT / "features" / "combined"

RESULT_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 37
TRAIN_SIZE = 2944
TEST_SIZE = 3669

CONFIGURATIONS = [
    {
        "name": "color",
        "directory": HANDCRAFTED_DIR,
        "features": 88,
        "k": 3,
        "distance": "euclidean"
    },
    {
        "name": "lbp",
        "directory": HANDCRAFTED_DIR,
        "features": 1003,
        "k": 3,
        "distance": "cosine"
    },
    {
        "name": "glcm",
        "directory": HANDCRAFTED_DIR,
        "features": 40,
        "k": 9,
        "distance": "euclidean"
    },
    {
        "name": "hog",
        "directory": HANDCRAFTED_DIR,
        "features": 8100,
        "k": 5,
        "distance": "cosine"
    },
    {
        "name": "pooled_hog",
        "directory": HANDCRAFTED_DIR,
        "features": 144,
        "k": 3,
        "distance": "cosine"
    },
    {
        "name": "handcrafted",
        "directory": HANDCRAFTED_DIR,
        "features": 9375,
        "k": 9,
        "distance": "cosine"
    },
    {
        "name": "bovw_200",
        "directory": BOVW_DIR,
        "features": 200,
        "k": 5,
        "distance": "cosine"
    },
    {
        "name": "all",
        "directory": COMBINED_DIR,
        "features": 10343,
        "k": 15,
        "distance": "cosine"
    }
]


def load_data(config):

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

    expected_dim = config["features"]

    if X_train.shape != (TRAIN_SIZE, expected_dim):
        raise ValueError(
            f"{name}: invalid TRAIN shape {X_train.shape}"
        )

    if X_test.shape != (TEST_SIZE, expected_dim):
        raise ValueError(
            f"{name}: invalid TEST shape {X_test.shape}"
        )

    if y_train.shape != (TRAIN_SIZE,):
        raise ValueError(f"{name}: invalid TRAIN labels")

    if y_test.shape != (TEST_SIZE,):
        raise ValueError(f"{name}: invalid TEST labels")

    # Verify all feature sets use identical dataset ordering.
    reference_train = np.load(
        ROOT / "data" / "splits" / "train_labels.npy"
    )

    reference_test = np.load(
        ROOT / "data" / "splits" / "test_labels.npy"
    )

    np.testing.assert_array_equal(
        y_train,
        reference_train
    )

    np.testing.assert_array_equal(
        y_test,
        reference_test
    )

    if not np.all(np.isfinite(X_train)):
        raise ValueError(f"{name}: non-finite TRAIN features")

    if not np.all(np.isfinite(X_test)):
        raise ValueError(f"{name}: non-finite TEST features")

    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred):

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    np.add.at(
        cm,
        (y_true.astype(int), y_pred.astype(int)),
        1
    )

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = cm[class_id, class_id]

        fp = cm[:, class_id].sum() - tp

        fn = cm[class_id, :].sum() - tp

        denominator = 2 * tp + fp + fn

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return accuracy, macro_f1, cm


def save_confusion_matrix(name, cm):

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
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"{name.upper()} - TEST Confusion Matrix"
    )

    ax.set_xlabel("Predicted Breed ID")
    ax.set_ylabel("True Breed ID")

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))

    ax.tick_params(
        axis="both",
        labelsize=6
    )

    ax.tick_params(
        axis="x",
        labelrotation=90
    )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"{name}_test_confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)


def run_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    k,
    distance
):

    model = KNNFromScratch(
        k=k,
        distance=distance
    )

    model.fit(
        X_train,
        y_train
    )

    start = time.perf_counter()

    predictions = model.predict(
        X_test,
        batch_size=32
    )

    elapsed = time.perf_counter() - start

    accuracy, macro_f1, cm = calculate_metrics(
        y_test,
        predictions
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "time_ms": elapsed / len(y_test) * 1000,
        "predictions": predictions,
        "confusion_matrix": cm
    }


def evaluate(config):

    name = config["name"]

    print("\n" + "=" * 75)
    print(f"FINAL TEST: {name.upper()}")
    print("=" * 75)

    X_train, y_train, X_test, y_test = load_data(
        config
    )

    print(f"TRAIN: {X_train.shape}")
    print(f"TEST:  {X_test.shape}")

    # Fit standardization on TRAIN only.
    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nRunning 1-NN Euclidean baseline...")

    baseline = run_knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=1,
        distance="euclidean"
    )

    print(
        f"1-NN Accuracy: "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"1-NN Macro-F1: "
        f"{baseline['macro_f1']:.4f}"
    )

    print("\nRunning selected K-NN configuration...")

    best = run_knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=config["k"],
        distance=config["distance"]
    )

    save_confusion_matrix(
        name,
        best["confusion_matrix"]
    )

    np.save(
        RESULT_DIR / f"{name}_test_predictions.npy",
        best["predictions"]
    )

    result = {
        "model": name,
        "features": config["features"],
        "k": config["k"],
        "distance": config["distance"],
        "one_nn_test_accuracy": baseline["accuracy"],
        "one_nn_test_macro_f1": baseline["macro_f1"],
        "test_accuracy": best["accuracy"],
        "test_macro_f1": best["macro_f1"],
        "correct_predictions": int(
            np.trace(best["confusion_matrix"])
        ),
        "test_images": len(y_test),
        "prediction_ms_per_image": best["time_ms"]
    }

    print("\nRESULTS")
    print("-" * 75)

    print(f"K: {config['k']}")
    print(f"Distance: {config['distance']}")

    print(
        f"TEST Accuracy: "
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        f"TEST Macro-F1: "
        f"{best['macro_f1']:.4f}"
    )

    print(
        f"Prediction Time: "
        f"{best['time_ms']:.3f} ms/image"
    )

    return result


def main():

    print("=" * 80)
    print("CSE480 - FINAL TEST EVALUATION")
    print("HANDCRAFTED + BOVW + ALL FEATURES")
    print("=" * 80)

    results = []

    for config in CONFIGURATIONS:

        result = evaluate(config)

        results.append(result)

    csv_path = (
        RESULT_DIR / "remaining_final_test_results.csv"
    )

    json_path = (
        RESULT_DIR / "remaining_final_test_results.json"
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 105)
    print("FINAL TEST COMPARISON")
    print("=" * 105)

    print(
        f"{'Model':<16}"
        f"{'Features':>10}"
        f"{'1-NN':>12}"
        f"{'K-NN':>12}"
        f"{'Macro-F1':>12}"
        f"{'K':>6}"
        f"{'Distance':>12}"
        f"{'Time(ms)':>12}"
    )

    print("-" * 105)

    for row in results:

        print(
            f"{row['model']:<16}"
            f"{row['features']:>10}"
            f"{row['one_nn_test_accuracy'] * 100:>11.2f}%"
            f"{row['test_accuracy'] * 100:>11.2f}%"
            f"{row['test_macro_f1']:>12.4f}"
            f"{row['k']:>6}"
            f"{row['distance']:>12}"
            f"{row['prediction_ms_per_image']:>12.3f}"
        )

    print("\nResults saved to:")
    print(csv_path)
    print(json_path)

    print("\nConfusion matrices saved to:")
    print(FIGURE_DIR)

    print("\n" + "=" * 80)
    print("ALL REMAINING TEST EVALUATIONS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
