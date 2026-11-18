import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


VAL_RESULTS = {
    "color": 13.59,
    "lbp": 17.53,
    "glcm": 9.38,
    "hog": 11.14,
    "pooled_hog": 10.05,
    "handcrafted": 14.81,
    "bovw_200": 11.68,
    "all": 54.62,
    "mrmr_20": 81.66,
    "mrmr_50": 88.45,
    "mrmr_100": 89.81,
    "transformer": 91.03
}


DISPLAY_NAMES = {
    "color": "Color",
    "lbp": "LBP",
    "glcm": "GLCM",
    "hog": "HOG",
    "pooled_hog": "Pooled HOG",
    "handcrafted": "All Handcrafted",
    "bovw_200": "BoVW-200",
    "all": "All Features",
    "mrmr_20": "mRMR-20",
    "mrmr_50": "mRMR-50",
    "mrmr_100": "mRMR-100",
    "transformer": "ViT-B/16"
}


ORDER = [
    "color",
    "lbp",
    "glcm",
    "hog",
    "pooled_hog",
    "handcrafted",
    "bovw_200",
    "all",
    "mrmr_20",
    "mrmr_50",
    "mrmr_100",
    "transformer"
]


def load_csv(path):

    with open(path, "r", newline="") as file:
        return list(csv.DictReader(file))


def load_test_results():

    results = {}

    paths = [
        TABLE_DIR / "remaining_final_test_results.csv",
        TABLE_DIR / "final_test_results.csv"
    ]

    for path in paths:

        if not path.exists():
            raise FileNotFoundError(path)

        for row in load_csv(path):

            name = row["model"]

            results[name] = {
                "features": int(row["features"]),
                "accuracy": float(row["test_accuracy"]) * 100,
                "macro_f1": float(row["test_macro_f1"]),
                "time_ms": float(
                    row["prediction_ms_per_image"]
                )
            }

    missing = set(ORDER) - set(results)

    if missing:
        raise ValueError(
            f"Missing TEST results: {sorted(missing)}"
        )

    return results


def plot_validation_vs_test(results):

    names = [
        DISPLAY_NAMES[name]
        for name in ORDER
    ]

    val = [
        VAL_RESULTS[name]
        for name in ORDER
    ]

    test = [
        results[name]["accuracy"]
        for name in ORDER
    ]

    positions = np.arange(len(ORDER))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    ax.bar(
        positions - width / 2,
        val,
        width,
        label="Validation"
    )

    ax.bar(
        positions + width / 2,
        test,
        width,
        label="Test"
    )

    ax.set_title(
        "Validation vs Test Accuracy",
        fontsize=16
    )

    ax.set_ylabel("Accuracy (%)")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        names,
        rotation=45,
        ha="right"
    )

    ax.set_ylim(0, 105)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    path = FIGURE_DIR / "final_val_vs_test.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def plot_accuracy_vs_features(results):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for name in ORDER:

        feature_count = results[name]["features"]
        accuracy = results[name]["accuracy"]

        ax.scatter(
            feature_count,
            accuracy,
            s=90
        )

        ax.annotate(
            DISPLAY_NAMES[name],
            (feature_count, accuracy),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_xscale("log")

    ax.set_title(
        "Test Accuracy vs Feature Dimension",
        fontsize=16
    )

    ax.set_xlabel(
        "Number of Features (Log Scale)"
    )

    ax.set_ylabel("Test Accuracy (%)")

    ax.set_ylim(0, 105)

    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = FIGURE_DIR / "accuracy_vs_features.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def plot_mrmr_tradeoff(results):

    selected = [
        "mrmr_20",
        "mrmr_50",
        "mrmr_100",
        "transformer"
    ]

    labels = [
        DISPLAY_NAMES[name]
        for name in selected
    ]

    features = [
        results[name]["features"]
        for name in selected
    ]

    accuracies = [
        results[name]["accuracy"]
        for name in selected
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        features,
        accuracies,
        marker="o",
        linewidth=2
    )

    for x, y, label in zip(
        features,
        accuracies,
        labels
    ):

        ax.annotate(
            label,
            (x, y),
            xytext=(7, 7),
            textcoords="offset points"
        )

    ax.set_title(
        "mRMR Dimensionality vs Test Accuracy",
        fontsize=15
    )

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Test Accuracy (%)")

    ax.set_xlim(0, 820)
    ax.set_ylim(75, 95)

    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = FIGURE_DIR / "mrmr_accuracy_tradeoff.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def plot_prediction_time(results):

    names = [
        DISPLAY_NAMES[name]
        for name in ORDER
    ]

    times = [
        results[name]["time_ms"]
        for name in ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    positions = np.arange(len(names))

    ax.barh(
        positions,
        times
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(names)

    ax.invert_yaxis()

    ax.set_xlabel(
        "K-NN Prediction Time (ms/image)"
    )

    ax.set_title(
        "Classification Computational Cost",
        fontsize=15
    )

    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    path = FIGURE_DIR / "knn_prediction_time.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def main():

    print("=" * 70)
    print("GENERATING FINAL RESULTS FIGURES")
    print("=" * 70)

    results = load_test_results()

    plot_validation_vs_test(results)

    plot_accuracy_vs_features(results)

    plot_mrmr_tradeoff(results)

    plot_prediction_time(results)

    print("\n" + "=" * 70)
    print("ALL FINAL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
