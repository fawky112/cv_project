import json
import hashlib
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = ROOT / "data" / "splits"
FEATURES_DIR = ROOT / "features"
RESULT_DIR = ROOT / "results" / "tables"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SIZES = {
    "train": 2944,
    "val": 736,
    "test": 3669
}

FEATURE_GROUPS = {
    "color": ("handcrafted", 88),
    "lbp": ("handcrafted", 1003),
    "glcm": ("handcrafted", 40),
    "hog": ("handcrafted", 8100),
    "pooled_hog": ("handcrafted", 144),
    "handcrafted": ("handcrafted", 9375),
    "bovw_200": ("bovw", 200),
    "transformer": ("transformer", 768),
    "all": ("combined", 10343),
    "mrmr_20": ("selected", 20),
    "mrmr_50": ("selected", 50),
    "mrmr_100": ("selected", 100),
}


def sha256_file(path):

    hasher = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def check_splits():

    print("\n" + "=" * 70)
    print("DATASET SPLIT VERIFICATION")
    print("=" * 70)

    split_data = {}
    all_names = set()

    for split, expected_size in EXPECTED_SIZES.items():

        names_path = SPLITS_DIR / f"{split}_names.npy"
        labels_path = SPLITS_DIR / f"{split}_labels.npy"

        names = np.load(names_path)
        labels = np.load(labels_path)

        assert len(names) == expected_size
        assert len(labels) == expected_size

        assert np.all((labels >= 0) & (labels < 37))

        current_names = set(map(str, names))

        assert len(current_names) == len(names), (
            f"Duplicate images within {split}"
        )

        overlap = all_names.intersection(current_names)

        assert not overlap, (
            f"Dataset leakage detected in {split}: "
            f"{len(overlap)} overlapping images"
        )

        all_names.update(current_names)

        split_data[split] = {
            "images": len(names),
            "classes": len(np.unique(labels)),
            "names_sha256": sha256_file(names_path),
            "labels_sha256": sha256_file(labels_path)
        }

        print(
            f"{split.upper():6s} | "
            f"Images: {len(names):4d} | "
            f"Classes: {len(np.unique(labels))}"
        )

    assert len(all_names) == 7349

    print("\nNo overlap between dataset splits.")

    return split_data


def check_feature_matrices():

    print("\n" + "=" * 70)
    print("FEATURE MATRIX VERIFICATION")
    print("=" * 70)

    results = {}

    for name, (folder, expected_dim) in FEATURE_GROUPS.items():

        directory = FEATURES_DIR / folder

        matrices = {}

        for split, expected_size in EXPECTED_SIZES.items():

            feature_path = (
                directory / f"{name}_{split}.npy"
            )

            labels_path = (
                directory / f"{name}_{split}_labels.npy"
            )

            X = np.load(
                feature_path,
                mmap_mode="r"
            )

            y = np.load(labels_path)

            reference_labels = np.load(
                SPLITS_DIR / f"{split}_labels.npy"
            )

            assert X.shape == (
                expected_size,
                expected_dim
            ), (
                f"{name}/{split}: "
                f"Unexpected shape {X.shape}"
            )

            np.testing.assert_array_equal(
                y,
                reference_labels
            )

            assert X.dtype == np.float32, (
                f"{name}/{split}: Unexpected dtype {X.dtype}"
            )

            matrices[split] = {
                "shape": list(X.shape),
                "dtype": str(X.dtype),
                "size_mb": round(
                    X.nbytes / 1024**2,
                    4
                )
            }

        results[name] = matrices

        print(
            f"{name:15s} | "
            f"Dim: {expected_dim:5d} | PASS"
        )

    return results


def check_mrmr():

    print("\n" + "=" * 70)
    print("MRMR FEATURE VERIFICATION")
    print("=" * 70)

    selected_dir = FEATURES_DIR / "selected"
    transformer_dir = FEATURES_DIR / "transformer"

    results = {}

    for k in [20, 50, 100]:

        name = f"mrmr_{k}"

        indices = np.load(
            selected_dir / f"{name}_indices.npy"
        ).astype(int)

        assert indices.shape == (k,)
        assert len(np.unique(indices)) == k

        # Transformer occupies columns 9375:10143
        assert np.all(indices >= 9375)
        assert np.all(indices < 9375 + 768)

        local_indices = indices - 9375

        max_difference = 0.0

        for split in ["train", "val", "test"]:

            X_transformer = np.load(
                transformer_dir / f"transformer_{split}.npy",
                mmap_mode="r"
            )

            X_selected = np.load(
                selected_dir / f"{name}_{split}.npy",
                mmap_mode="r"
            )

            expected = X_transformer[:, local_indices]

            difference = float(
                np.max(
                    np.abs(expected - X_selected)
                )
            )

            assert np.allclose(
                expected,
                X_selected,
                rtol=1e-5,
                atol=1e-6
            ), (
                f"{name}/{split}: "
                f"Selected features do not match ViT columns"
            )

            max_difference = max(
                max_difference,
                difference
            )

        results[name] = {
            "selected_features": k,
            "source": "transformer",
            "maximum_absolute_difference": max_difference,
            "indices_sha256": sha256_file(
                selected_dir / f"{name}_indices.npy"
            )
        }

        print(
            f"{name:12s} | "
            f"Features: {k:3d} | "
            f"Max Difference: {max_difference:.8f} | PASS"
        )

    return results


def check_dictionaries():

    print("\n" + "=" * 70)
    print("FEATURE DICTIONARY FILES")
    print("=" * 70)

    dictionary_files = sorted(
        FEATURES_DIR.rglob("*dictionary.json")
    )

    results = []

    for path in dictionary_files:

        with open(path, "r") as file:
            dictionary = json.load(file)

        if isinstance(dictionary, (dict, list)):
            entries = len(dictionary)
        else:
            raise ValueError(
                f"Unexpected dictionary format: {path}"
            )

        relative_path = str(path.relative_to(ROOT))

        results.append({
            "file": relative_path,
            "entries": entries
        })

        print(
            f"{relative_path} | "
            f"Entries: {entries}"
        )

    if not dictionary_files:
        raise FileNotFoundError(
            "No feature dictionary files found"
        )

    return results


def get_versions():

    packages = [
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "Pillow",
        "mrmr-selection"
    ]

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform()
    }

    for package in packages:

        try:
            versions[package] = version(package)

        except PackageNotFoundError:
            versions[package] = "Not found"

    try:
        import torch

        versions["cuda_available"] = torch.cuda.is_available()
        versions["torch_cuda"] = torch.version.cuda

        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)

    except ImportError:
        versions["cuda_available"] = False

    print("\n" + "=" * 70)
    print("SOFTWARE ENVIRONMENT")
    print("=" * 70)

    for key, value in versions.items():
        print(f"{key:20s}: {value}")

    return versions


def main():

    print("=" * 70)
    print("CSE480 - FINAL PROJECT AUDIT")
    print("=" * 70)

    audit = {}

    audit["dataset"] = check_splits()
    audit["feature_matrices"] = check_feature_matrices()
    audit["mrmr"] = check_mrmr()
    audit["feature_dictionaries"] = check_dictionaries()
    audit["environment"] = get_versions()

    output_path = (
        RESULT_DIR / "project_audit.json"
    )

    with open(output_path, "w") as file:

        json.dump(
            audit,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("PROJECT AUDIT PASSED")
    print("=" * 70)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
