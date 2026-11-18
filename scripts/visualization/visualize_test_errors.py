from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_DIR = ROOT / "data" / "splits"

RESULT_DIR = ROOT / "results" / "tables"

FIGURE_DIR = ROOT / "results" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 37

MODEL_NAME = "transformer"


def load_breed_names():

    annotation_path = (
        DATA_ROOT / "annotations" / "list.txt"
    )

    breed_names = {}

    with open(annotation_path, "r") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            image_name = parts[0]
            class_id = int(parts[1]) - 1

            breed_name = image_name.rsplit("_", 1)[0]

            if class_id in breed_names:
                if breed_names[class_id] != breed_name:
                    raise ValueError(
                        f"Inconsistent breed name: {class_id}"
                    )

            breed_names[class_id] = breed_name

    if len(breed_names) != NUM_CLASSES:
        raise ValueError("Expected 37 breeds")

    return breed_names


def load_test_data():

    names = np.load(
        SPLIT_DIR / "test_names.npy"
    )

    labels = np.load(
        SPLIT_DIR / "test_labels.npy"
    )

    predictions = np.load(
        RESULT_DIR / f"{MODEL_NAME}_test_predictions.npy"
    )

    if not (
        len(names)
        == len(labels)
        == len(predictions)
        == 3669
    ):
        raise ValueError(
            "TEST arrays have inconsistent lengths"
        )

    saved_labels = np.load(
        ROOT
        / "features"
        / "transformer"
        / "transformer_test_labels.npy"
    )

    np.testing.assert_array_equal(
        labels,
        saved_labels
    )

    return names, labels, predictions


def find_top_confusions(y_true, y_pred, top_n=6):

    confusion_pairs = {}

    for true_label, predicted_label in zip(
        y_true,
        y_pred
    ):

        true_label = int(true_label)
        predicted_label = int(predicted_label)

        if true_label == predicted_label:
            continue

        pair = (
            true_label,
            predicted_label
        )

        confusion_pairs[pair] = (
            confusion_pairs.get(pair, 0) + 1
        )

    ranked = sorted(
        confusion_pairs.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    return ranked[:top_n]


def select_examples(
    names,
    labels,
    predictions,
    top_pairs,
    examples_per_pair=2
):

    selected = []

    for (true_id, predicted_id), count in top_pairs:

        matching_indices = np.flatnonzero(
            (labels == true_id)
            & (predictions == predicted_id)
        )

        for image_index in matching_indices[:examples_per_pair]:

            selected.append({
                "index": int(image_index),
                "name": str(names[image_index]),
                "true_id": int(true_id),
                "predicted_id": int(predicted_id),
                "pair_count": int(count)
            })

    return selected


def make_figure(examples, breed_names):

    columns = 3
    rows = int(np.ceil(len(examples) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 4.6)
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, example in zip(axes, examples):

        image_path = (
            DATA_ROOT
            / "images"
            / f"{example['name']}.jpg"
        )

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB").copy()

        ax.imshow(image_rgb)

        true_name = breed_names[example["true_id"]]
        predicted_name = breed_names[example["predicted_id"]]

        ax.set_title(
            f"True: {true_name}\n"
            f"Predicted: {predicted_name}",
            fontsize=9
        )

        ax.axis("off")

    for ax in axes[len(examples):]:
        ax.axis("off")

    fig.suptitle(
        "ViT-B/16 + K-NN: TEST Misclassification Examples",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    output_path = (
        FIGURE_DIR / "transformer_misclassifications.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


def main():

    print("=" * 70)
    print("VISUALIZING TRANSFORMER TEST ERRORS")
    print("=" * 70)

    breed_names = load_breed_names()

    names, labels, predictions = load_test_data()

    total_errors = int(
        np.sum(labels != predictions)
    )

    print(f"TEST Images: {len(names)}")
    print(f"Total Errors: {total_errors}")

    top_pairs = find_top_confusions(
        labels,
        predictions,
        top_n=6
    )

    print("\nTOP CONFUSION PAIRS")

    for (true_id, predicted_id), count in top_pairs:

        print(
            f"{breed_names[true_id]:25s} -> "
            f"{breed_names[predicted_id]:25s} | "
            f"{count}"
        )

    examples = select_examples(
        names,
        labels,
        predictions,
        top_pairs,
        examples_per_pair=2
    )

    output_path = make_figure(
        examples,
        breed_names
    )

    print(f"\nVisualized Images: {len(examples)}")
    print(f"Figure saved: {output_path}")

    print("\nVISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()
