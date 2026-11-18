import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mrmr import mrmr_classif


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "features" / "combined"
OUTPUT_DIR = ROOT / "features" / "selected"

SELECTION_SIZES = [20, 50, 100]

MAX_FEATURES = max(SELECTION_SIZES)

N_JOBS = 2


def load_data():

    print("Loading combined TRAIN and VAL features...")

    X_train = np.load(
        INPUT_DIR / "all_train.npy",
        mmap_mode="r"
    )

    X_val = np.load(
        INPUT_DIR / "all_val.npy",
        mmap_mode="r"
    )

    y_train = np.load(
        INPUT_DIR / "all_train_labels.npy"
    )

    y_val = np.load(
        INPUT_DIR / "all_val_labels.npy"
    )

    with open(
        INPUT_DIR / "all_feature_dictionary.json",
        "r"
    ) as file:
        dictionary = json.load(file)

    if X_train.shape[1] != len(dictionary):
        raise ValueError(
            "Feature matrix and dictionary dimensions differ"
        )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "TRAIN and VAL feature dimensions differ"
        )

    if len(X_train) != len(y_train):
        raise ValueError("TRAIN labels mismatch")

    if len(X_val) != len(y_val):
        raise ValueError("VAL labels mismatch")

    if not np.all(np.isfinite(X_train)):
        raise ValueError("Invalid TRAIN features")

    if not np.all(np.isfinite(X_val)):
        raise ValueError("Invalid VAL features")

    print(f"TRAIN: {X_train.shape}")
    print(f"VAL:   {X_val.shape}")

    return X_train, y_train, X_val, y_val, dictionary


def build_dataframe(X_train, y_train):

    print("\nPreparing TRAIN DataFrame...")

    # Keep unique names and preserve exact column positions.
    column_names = [
        f"feature_{index}"
        for index in range(X_train.shape[1])
    ]

    X_df = pd.DataFrame(
        np.asarray(X_train),
        columns=column_names
    )

    y_series = pd.Series(
        y_train,
        name="breed"
    )

    return X_df, y_series


def select_features(X_df, y_series):

    print("\n" + "=" * 70)
    print("RUNNING mRMR ON TRAIN ONLY")
    print("=" * 70)

    print(f"Candidate Features: {X_df.shape[1]}")
    print(f"Requested Features: {MAX_FEATURES}")
    print(f"Workers: {N_JOBS}")

    start = time.perf_counter()

    selected_names = mrmr_classif(
        X=X_df,
        y=y_series,
        K=MAX_FEATURES,
        relevance="f",
        redundancy="c",
        denominator="mean",
        n_jobs=N_JOBS,
        show_progress=True
    )

    elapsed = time.perf_counter() - start

    if len(selected_names) != MAX_FEATURES:
        raise RuntimeError(
            f"Expected {MAX_FEATURES} features, "
            f"got {len(selected_names)}"
        )

    selected_indices = np.array(
        [
            int(name.removeprefix("feature_"))
            for name in selected_names
        ],
        dtype=np.int64
    )

    if len(np.unique(selected_indices)) != MAX_FEATURES:
        raise RuntimeError("Duplicate selected features")

    if (
        selected_indices.min() < 0
        or selected_indices.max() >= X_df.shape[1]
    ):
        raise RuntimeError("Invalid selected column indices")

    print(f"\nmRMR completed in {elapsed:.2f} seconds")

    return selected_indices, elapsed


def save_selected_features(
    X_train,
    y_train,
    X_val,
    y_val,
    dictionary,
    ranked_indices,
    selection_time
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "mrmr_ranked_indices_100.npy",
        ranked_indices
    )

    summary = {}

    for k in SELECTION_SIZES:

        indices = ranked_indices[:k]

        X_train_selected = np.asarray(
            X_train[:, indices],
            dtype=np.float32
        )

        X_val_selected = np.asarray(
            X_val[:, indices],
            dtype=np.float32
        )

        if X_train_selected.shape != (len(y_train), k):
            raise RuntimeError("Unexpected TRAIN shape")

        if X_val_selected.shape != (len(y_val), k):
            raise RuntimeError("Unexpected VAL shape")

        np.save(
            OUTPUT_DIR / f"mrmr_{k}_train.npy",
            X_train_selected
        )

        np.save(
            OUTPUT_DIR / f"mrmr_{k}_val.npy",
            X_val_selected
        )

        np.save(
            OUTPUT_DIR / f"mrmr_{k}_train_labels.npy",
            y_train
        )

        np.save(
            OUTPUT_DIR / f"mrmr_{k}_val_labels.npy",
            y_val
        )

        np.save(
            OUTPUT_DIR / f"mrmr_{k}_indices.npy",
            indices
        )

        selected_dictionary = []

        for rank, original_index in enumerate(indices):

            entry = dict(
                dictionary[int(original_index)]
            )

            entry["selected_rank"] = rank + 1
            entry["original_column_index"] = int(original_index)

            selected_dictionary.append(entry)

        dictionary_path = (
            OUTPUT_DIR / f"mrmr_{k}_dictionary.json"
        )

        with open(dictionary_path, "w") as file:

            json.dump(
                selected_dictionary,
                file,
                indent=2
            )

        family_counts = Counter(
            entry["feature_group"]
            for entry in selected_dictionary
        )

        summary[str(k)] = {
            "selected_features": k,
            "train_shape": list(X_train_selected.shape),
            "val_shape": list(X_val_selected.shape),
            "family_counts": dict(family_counts)
        }

        print("\n" + "-" * 60)
        print(f"SELECTED TOP {k} FEATURES")
        print("-" * 60)

        print(f"TRAIN: {X_train_selected.shape}")
        print(f"VAL:   {X_val_selected.shape}")

        print("Feature distribution:")

        for family, count in family_counts.most_common():

            print(f"  {family:15s}: {count}")

    metadata = {
        "algorithm": "mRMR",
        "implementation": "mrmr-selection",
        "relevance": "F-statistic",
        "redundancy": "Pearson correlation",
        "denominator": "mean",
        "fit_split": "TRAIN",
        "selection_time_seconds": selection_time,
        "total_candidate_features": int(X_train.shape[1]),
        "selection_sizes": SELECTION_SIZES,
        "selected_indices_order": "mRMR ranking",
        "configurations": summary
    }

    with open(
        OUTPUT_DIR / "mrmr_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


def main():

    print("=" * 70)
    print("mRMR FEATURE SELECTION")
    print("=" * 70)

    X_train, y_train, X_val, y_val, dictionary = load_data()

    X_df, y_series = build_dataframe(
        X_train,
        y_train
    )

    ranked_indices, elapsed = select_features(
        X_df,
        y_series
    )

    save_selected_features(
        X_train,
        y_train,
        X_val,
        y_val,
        dictionary,
        ranked_indices,
        elapsed
    )

    print("\n" + "=" * 70)
    print("mRMR FEATURE SELECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
