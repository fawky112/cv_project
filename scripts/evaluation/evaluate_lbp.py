import csv
import time
from pathlib import Path

import numpy as np

from src.classifiers.knn import (
    StandardScalerScratch,
    KNNFromScratch
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "features" / "handcrafted"

RESULT_ROOT = PROJECT_ROOT / "results" / "tables"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def accuracy_score(y_true, y_pred):

    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true, y_pred, num_classes=37):

    scores = []

    for class_id in range(num_classes):

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


def evaluate(X_train, y_train, X_val, y_val, k, distance):

    model = KNNFromScratch(
        k=k,
        distance=distance
    )

    model.fit(X_train, y_train)

    start = time.perf_counter()

    predictions = model.predict(
        X_val,
        batch_size=64
    )

    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = macro_f1_score(
        y_val,
        predictions
    )

    return {
        "feature_set": "lbp",
        "k": k,
        "distance": distance,
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "prediction_time_ms": (
            elapsed / len(y_val) * 1000
        )
    }


def main():

    X_train = np.load(
        FEATURE_ROOT / "lbp_train.npy"
    )

    y_train = np.load(
        FEATURE_ROOT / "lbp_train_labels.npy"
    )

    X_val = np.load(
        FEATURE_ROOT / "lbp_val.npy"
    )

    y_val = np.load(
        FEATURE_ROOT / "lbp_val_labels.npy"
    )

    print("=" * 75)
    print("LBP FEATURE CLASSIFICATION")
    print("=" * 75)

    print(f"TRAIN: {X_train.shape}")
    print(f"VAL:   {X_val.shape}")

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    results = []

    print("\n1-NN BASELINE")
    print("-" * 75)

    baseline = evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        k=1,
        distance="euclidean"
    )

    results.append(baseline)

    print(
        f"Accuracy: "
        f"{baseline['val_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro-F1: "
        f"{baseline['val_macro_f1']:.4f}"
    )

    print("\nK-NN GRID SEARCH")
    print("-" * 75)

    for distance in ["euclidean", "cosine"]:

        for k in [3, 5, 9, 15]:

            result = evaluate(
                X_train,
                y_train,
                X_val,
                y_val,
                k=k,
                distance=distance
            )

            results.append(result)

            print(
                f"K={k:<2} | "
                f"Distance={distance:<9} | "
                f"Accuracy={result['val_accuracy'] * 100:6.2f}% | "
                f"Macro-F1={result['val_macro_f1']:.4f} | "
                f"Time={result['prediction_time_ms']:.3f} ms/image"
            )

    best = max(
        results[1:],
        key=lambda r: r["val_accuracy"]
    )

    print("\n" + "=" * 75)
    print("BEST K-NN CONFIGURATION")
    print("=" * 75)

    print(f"K: {best['k']}")

    print(f"Distance: {best['distance']}")

    print(
        f"VAL Accuracy: "
        f"{best['val_accuracy'] * 100:.2f}%"
    )

    print(
        f"VAL Macro-F1: "
        f"{best['val_macro_f1']:.4f}"
    )

    output_path = RESULT_ROOT / "lbp_knn_grid.csv"

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
