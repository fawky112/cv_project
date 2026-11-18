import csv
import time
from pathlib import Path

import numpy as np

from src.classifiers.knn import (
    StandardScalerScratch,
    KNNFromScratch
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = PROJECT_ROOT / "features" / "combined"
RESULT_ROOT = PROJECT_ROOT / "results" / "tables"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = ["all"]

K_VALUES = [3, 5, 9, 15]
DISTANCES = ["euclidean", "cosine"]


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true, y_pred):

    f1_scores = []

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

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_set,
    k,
    distance
):

    model = KNNFromScratch(
        k=k,
        distance=distance
    )

    model.fit(X_train, y_train)

    start = time.perf_counter()

    predictions = model.predict(
        X_val,
        batch_size=32
    )

    elapsed = time.perf_counter() - start

    return {
        "feature_set": feature_set,
        "k": k,
        "distance": distance,
        "val_accuracy": accuracy_score(y_val, predictions),
        "val_macro_f1": macro_f1_score(y_val, predictions),
        "prediction_time_ms": elapsed / len(y_val) * 1000
    }


def evaluate_feature_set(feature_set):

    print("\n" + "=" * 75)
    print(f"EVALUATING: {feature_set.upper()}")
    print("=" * 75)

    X_train = np.load(
        FEATURE_ROOT / f"{feature_set}_train.npy"
    )

    y_train = np.load(
        FEATURE_ROOT / f"{feature_set}_train_labels.npy"
    )

    X_val = np.load(
        FEATURE_ROOT / f"{feature_set}_val.npy"
    )

    y_val = np.load(
        FEATURE_ROOT / f"{feature_set}_val_labels.npy"
    )

    print(f"TRAIN: {X_train.shape}")
    print(f"VAL:   {X_val.shape}")

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    results = []

    baseline = evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_set,
        k=1,
        distance="euclidean"
    )

    results.append(baseline)

    print("\n1-NN BASELINE")
    print(
        f"Accuracy: {baseline['val_accuracy'] * 100:.2f}%"
    )
    print(
        f"Macro-F1: {baseline['val_macro_f1']:.4f}"
    )

    print("\nK-NN GRID SEARCH")

    for distance in DISTANCES:

        for k in K_VALUES:

            result = evaluate(
                X_train,
                y_train,
                X_val,
                y_val,
                feature_set,
                k,
                distance
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
        key=lambda r: (
            r["val_accuracy"],
            r["val_macro_f1"]
        )
    )

    print("\nBEST K-NN CONFIGURATION")

    print(f"K: {best['k']}")
    print(f"Distance: {best['distance']}")
    print(
        f"VAL Accuracy: {best['val_accuracy'] * 100:.2f}%"
    )
    print(
        f"VAL Macro-F1: {best['val_macro_f1']:.4f}"
    )

    output_path = (
        RESULT_ROOT / f"{feature_set}_knn_grid.csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")

    return baseline, best


def main():

    summary = []

    for feature_set in FEATURE_SETS:

        baseline, best = evaluate_feature_set(
            feature_set
        )

        summary.append({
            "feature_set": feature_set,
            "one_nn_val_accuracy": baseline["val_accuracy"],
            "best_knn_val_accuracy": best["val_accuracy"],
            "best_knn_macro_f1": best["val_macro_f1"],
            "best_k": best["k"],
            "best_distance": best["distance"]
        })

    output_path = (
        RESULT_ROOT / "all_features_comparison.csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 75)
    print("FINAL ALL FEATURES COMPARISON")
    print("=" * 75)

    for row in summary:

        print(
            f"{row['feature_set']:12s} | "
            f"1-NN: {row['one_nn_val_accuracy'] * 100:.2f}% | "
            f"Best K-NN: {row['best_knn_val_accuracy'] * 100:.2f}% | "
            f"K={row['best_k']} | "
            f"{row['best_distance']}"
        )


if __name__ == "__main__":
    main()
