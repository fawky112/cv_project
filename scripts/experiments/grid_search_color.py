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
RESULTS_ROOT = PROJECT_ROOT / "results" / "tables"

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 9, 15]
DISTANCES = ["euclidean", "cosine"]


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true, y_pred, num_classes=37):

    scores = []

    for class_id in range(num_classes):

        tp = np.sum((y_true == class_id) & (y_pred == class_id))
        fp = np.sum((y_true != class_id) & (y_pred == class_id))
        fn = np.sum((y_true == class_id) & (y_pred != class_id))

        denominator = 2 * tp + fp + fn

        f1 = 2 * tp / denominator if denominator > 0 else 0.0

        scores.append(f1)

    return float(np.mean(scores))


def main():

    X_train = np.load(FEATURE_ROOT / "color_train.npy")
    y_train = np.load(FEATURE_ROOT / "color_train_labels.npy")

    X_val = np.load(FEATURE_ROOT / "color_val.npy")
    y_val = np.load(FEATURE_ROOT / "color_val_labels.npy")

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    results = []

    print("=" * 75)
    print("COLOR FEATURES - KNN GRID SEARCH")
    print("=" * 75)

    print(f"TRAIN: {X_train.shape}")
    print(f"VAL:   {X_val.shape}")

    for distance in DISTANCES:

        for k in K_VALUES:

            model = KNNFromScratch(
                k=k,
                distance=distance
            )

            model.fit(X_train, y_train)

            start = time.perf_counter()

            predictions = model.predict(X_val)

            elapsed = time.perf_counter() - start

            accuracy = accuracy_score(y_val, predictions)

            macro_f1 = macro_f1_score(y_val, predictions)

            time_per_image = elapsed / len(X_val) * 1000

            results.append({
                "feature_set": "color",
                "k": k,
                "distance": distance,
                "val_accuracy": accuracy,
                "val_macro_f1": macro_f1,
                "total_time_sec": elapsed,
                "time_per_image_ms": time_per_image
            })

            print(
                f"K={k:<2} | "
                f"Distance={distance:<9} | "
                f"Accuracy={accuracy * 100:6.2f}% | "
                f"Macro-F1={macro_f1:.4f} | "
                f"Time={time_per_image:.3f} ms/image"
            )

    best = max(
        results,
        key=lambda row: row["val_accuracy"]
    )

    print("\n" + "=" * 75)
    print("BEST CONFIGURATION")
    print("=" * 75)

    print(f"K: {best['k']}")
    print(f"Distance: {best['distance']}")
    print(f"VAL Accuracy: {best['val_accuracy'] * 100:.2f}%")
    print(f"VAL Macro-F1: {best['val_macro_f1']:.4f}")

    output_path = RESULTS_ROOT / "color_knn_grid.csv"

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
