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


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def macro_f1_score(y_true, y_pred, num_classes=37):

    f1_scores = []

    for class_id in range(num_classes):

        tp = np.sum(
            (y_pred == class_id) &
            (y_true == class_id)
        )

        fp = np.sum(
            (y_pred == class_id) &
            (y_true != class_id)
        )

        fn = np.sum(
            (y_pred != class_id) &
            (y_true == class_id)
        )

        denominator = 2 * tp + fp + fn

        if denominator > 0:
            f1 = (2 * tp) / denominator
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return np.mean(f1_scores)


def main():

    print("=" * 60)
    print("COLOR FEATURE CLASSIFICATION")
    print("=" * 60)

    X_train = np.load(FEATURE_ROOT / "color_train.npy")
    y_train = np.load(FEATURE_ROOT / "color_train_labels.npy")

    X_val = np.load(FEATURE_ROOT / "color_val.npy")
    y_val = np.load(FEATURE_ROOT / "color_val_labels.npy")

    print(f"TRAIN Features: {X_train.shape}")
    print(f"VAL Features:   {X_val.shape}")

    print("\nApplying Standardization...")

    scaler = StandardScalerScratch()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("Training Statistics Applied Successfully!")

    print("\nRunning 1-NN Classification...")

    model = KNNFromScratch(
        k=1,
        distance="euclidean"
    )

    model.fit(X_train, y_train)

    start = time.perf_counter()

    predictions = model.predict(X_val)

    elapsed = time.perf_counter() - start

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = macro_f1_score(y_val, predictions)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"1-NN VAL Accuracy: {accuracy * 100:.2f}%")
    print(f"1-NN VAL Macro-F1: {macro_f1:.4f}")

    print(f"Total Prediction Time: {elapsed:.3f} sec")

    print(
        f"Prediction Time/Image: "
        f"{elapsed / len(X_val) * 1000:.3f} ms"
    )

    results = np.array([
        ["Feature Set", "Classifier", "Accuracy", "Macro-F1"],
        ["Color", "1-NN", f"{accuracy:.6f}", f"{macro_f1:.6f}"]
    ])

    np.savetxt(
        RESULTS_ROOT / "color_1nn_results.csv",
        results,
        delimiter=",",
        fmt="%s"
    )

    print("\nResults saved to:")
    print(RESULTS_ROOT / "color_1nn_results.csv")


if __name__ == "__main__":
    main()
