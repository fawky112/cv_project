import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "features"
TABLE_DIR = ROOT / "results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


CONFIGURATIONS = {
    "color": "handcrafted",
    "lbp": "handcrafted",
    "glcm": "handcrafted",
    "hog": "handcrafted",
    "pooled_hog": "handcrafted",
    "handcrafted": "handcrafted",
    "bovw_200": "bovw",
    "all": "combined",
    "mrmr_20": "selected",
    "mrmr_50": "selected",
    "mrmr_100": "selected",
    "transformer": "transformer",
}


def load_json(path):

    with open(path, "r") as file:
        return json.load(file)


def load_results():

    results = {}

    files = [
        TABLE_DIR / "remaining_final_test_results.csv",
        TABLE_DIR / "final_test_results.csv",
    ]

    for path in files:

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                name = row["model"]

                results[name] = {
                    "accuracy": float(row["test_accuracy"]) * 100,
                    "macro_f1": float(row["test_macro_f1"]),
                    "prediction_ms": float(
                        row["prediction_ms_per_image"]
                    ),
                }

    return results


def load_extraction_times():

    path = (
        TABLE_DIR
        / "remaining_test_extraction_times.json"
    )

    if not path.exists():
        return {}

    data = load_json(path)

    return data.get("ms_per_image", {})


def get_feature_info(name, directory_name):

    directory = FEATURE_DIR / directory_name

    train_path = directory / f"{name}_train.npy"
    test_path = directory / f"{name}_test.npy"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    X_train = np.load(
        train_path,
        mmap_mode="r"
    )

    X_test = np.load(
        test_path,
        mmap_mode="r"
    )

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"{name}: expected 2D feature matrices"
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{name}: feature dimension mismatch"
        )

    return {
        "features": int(X_train.shape[1]),
        "dtype": str(X_train.dtype),
        "bytes_per_image": int(
            X_train.shape[1] * X_train.dtype.itemsize
        ),
        "train_matrix_mb": (
            X_train.nbytes / (1024 ** 2)
        ),
        "test_matrix_mb": (
            X_test.nbytes / (1024 ** 2)
        ),
        "train_file_mb": (
            train_path.stat().st_size / (1024 ** 2)
        ),
        "test_file_mb": (
            test_path.stat().st_size / (1024 ** 2)
        ),
    }


def main():

    print("=" * 105)
    print("CSE480 - COMPUTATIONAL COST & MEMORY ANALYSIS")
    print("=" * 105)

    results = load_results()
    extraction_times = load_extraction_times()

    rows = []

    for name, directory_name in CONFIGURATIONS.items():

        if name not in results:
            raise ValueError(
                f"Missing TEST results for {name}"
            )

        feature_info = get_feature_info(
            name,
            directory_name
        )

        result = results[name]

        row = {
            "model": name,
            "features": feature_info["features"],
            "dtype": feature_info["dtype"],
            "feature_bytes_per_image": (
                feature_info["bytes_per_image"]
            ),
            "feature_kb_per_image": (
                feature_info["bytes_per_image"] / 1024
            ),
            "train_matrix_mb": (
                feature_info["train_matrix_mb"]
            ),
            "test_matrix_mb": (
                feature_info["test_matrix_mb"]
            ),
            "train_npy_file_mb": (
                feature_info["train_file_mb"]
            ),
            "test_npy_file_mb": (
                feature_info["test_file_mb"]
            ),
            "extraction_ms_per_image": (
                extraction_times.get(name, "")
            ),
            "knn_prediction_ms_per_image": (
                result["prediction_ms"]
            ),
            "test_accuracy_percent": (
                result["accuracy"]
            ),
            "test_macro_f1": (
                result["macro_f1"]
            ),
        }

        rows.append(row)

    output_csv = (
        TABLE_DIR / "computational_cost_comparison.csv"
    )

    with open(output_csv, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{'Model':<17}"
        f"{'Features':>10}"
        f"{'KB/Image':>12}"
        f"{'TRAIN MB':>12}"
        f"{'TEST MB':>12}"
        f"{'KNN ms':>12}"
        f"{'Accuracy':>12}"
    )

    print("-" * 100)

    for row in rows:

        print(
            f"{row['model']:<17}"
            f"{row['features']:>10}"
            f"{row['feature_kb_per_image']:>12.2f}"
            f"{row['train_matrix_mb']:>12.2f}"
            f"{row['test_matrix_mb']:>12.2f}"
            f"{row['knn_prediction_ms_per_image']:>12.3f}"
            f"{row['test_accuracy_percent']:>11.2f}%"
        )

    print("\n" + "=" * 105)
    print("FEATURE EXTRACTION TIME")
    print("=" * 105)

    for name, ms in extraction_times.items():

        print(
            f"{name:<17} | "
            f"{float(ms):>8.3f} ms/image"
        )

    print("\nImportant notes:")
    print(
        "- Matrix memory represents saved feature values, "
        "not peak process RAM."
    )
    print(
        "- K-NN timing excludes feature extraction."
    )
    print(
        "- Extraction timings exclude image loading "
        "and resizing."
    )
    print(
        "- HOG and pooled HOG share extraction computation."
    )
    print(
        "- mRMR features still require the full "
        "ViT embedding to be extracted."
    )
    print(
        "- Composite and ViT extraction timings are "
        "not estimated without matching measurements."
    )

    print(f"\nSaved: {output_csv}")

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
