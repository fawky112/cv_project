from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TRANSFORMER_DIR = ROOT / "features" / "transformer"

SELECTED_DIR = ROOT / "features" / "selected"

HANDCRAFTED_DIM = 9375
TRANSFORMER_DIM = 768


def main():

    X_test = np.load(
        TRANSFORMER_DIR / "transformer_test.npy"
    )

    y_test = np.load(
        TRANSFORMER_DIR / "transformer_test_labels.npy"
    )

    if X_test.shape != (3669, TRANSFORMER_DIM):
        raise ValueError(
            f"Unexpected TEST shape: {X_test.shape}"
        )

    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for k in [20, 50, 100]:

        global_indices = np.load(
            SELECTED_DIR / f"mrmr_{k}_indices.npy"
        )

        if len(global_indices) != k:
            raise ValueError(
                f"Invalid number of indices for mRMR-{k}"
            )

        start = HANDCRAFTED_DIM
        end = HANDCRAFTED_DIM + TRANSFORMER_DIM

        if not np.all(
            (global_indices >= start)
            & (global_indices < end)
        ):
            raise ValueError(
                f"mRMR-{k} includes non-Transformer features"
            )

        local_indices = global_indices - start

        X_selected = X_test[:, local_indices]

        np.save(
            SELECTED_DIR / f"mrmr_{k}_test.npy",
            X_selected.astype(np.float32)
        )

        np.save(
            SELECTED_DIR / f"mrmr_{k}_test_labels.npy",
            y_test
        )

        print(
            f"mRMR-{k}: TEST {X_selected.shape}"
        )

    print("\nMRMR TEST FEATURES PREPARED SUCCESSFULLY")


if __name__ == "__main__":
    main()
