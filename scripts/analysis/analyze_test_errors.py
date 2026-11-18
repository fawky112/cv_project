import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS = (
    Path.home()
    / "Downloads"
    / "oxford-iiit-pet"
    / "annotations"
    / "list.txt"
)

TABLE_DIR = ROOT / "results" / "tables"

MODELS = [
    "transformer",
    "mrmr_20",
    "mrmr_50",
    "mrmr_100"
]

NUM_CLASSES = 37


def load_breed_names():

    breed_names = {}

    with open(ANNOTATIONS, "r") as file:

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
                        f"Inconsistent breed for class {class_id}"
                    )

            breed_names[class_id] = breed_name

    if len(breed_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} breeds, "
            f"found {len(breed_names)}"
        )

    return breed_names


def analyze_model(model_name, breed_names):

    cm_path = (
        TABLE_DIR
        / f"{model_name}_test_confusion_matrix.npy"
    )

    cm = np.load(cm_path)

    if cm.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError("Invalid confusion matrix shape")

    total = cm.sum()

    correct = np.trace(cm)

    print("\n" + "=" * 75)
    print(f"TEST ERROR ANALYSIS: {model_name.upper()}")
    print("=" * 75)

    print(f"Total Images: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

    per_breed = []

    for class_id in range(NUM_CLASSES):

        support = int(cm[class_id].sum())

        correct_breed = int(cm[class_id, class_id])

        errors = support - correct_breed

        accuracy = (
            correct_breed / support
            if support > 0
            else 0.0
        )

        per_breed.append({
            "model": model_name,
            "class_id": class_id,
            "breed": breed_names[class_id],
            "test_images": support,
            "correct": correct_breed,
            "errors": errors,
            "breed_accuracy": accuracy
        })

    output_path = (
        TABLE_DIR / f"{model_name}_per_breed.csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=per_breed[0].keys()
        )

        writer.writeheader()
        writer.writerows(per_breed)

    worst = sorted(
        per_breed,
        key=lambda row: (
            row["breed_accuracy"],
            -row["test_images"]
        )
    )

    print("\n10 BREEDS WITH LOWEST ACCURACY")
    print("-" * 75)

    for row in worst[:10]:

        print(
            f"{row['breed']:25s} | "
            f"Accuracy: {row['breed_accuracy'] * 100:6.2f}% | "
            f"Errors: {row['errors']:3d} | "
            f"Images: {row['test_images']:3d}"
        )

    confusions = []

    for true_id in range(NUM_CLASSES):

        for predicted_id in range(NUM_CLASSES):

            if true_id == predicted_id:
                continue

            count = int(cm[true_id, predicted_id])

            if count > 0:

                confusions.append({
                    "model": model_name,
                    "true_breed": breed_names[true_id],
                    "predicted_breed": breed_names[predicted_id],
                    "count": count
                })

    confusions.sort(
        key=lambda row: row["count"],
        reverse=True
    )

    print("\nTOP 10 BREED CONFUSIONS")
    print("-" * 75)

    for row in confusions[:10]:

        print(
            f"{row['true_breed']:22s} -> "
            f"{row['predicted_breed']:22s} | "
            f"{row['count']} images"
        )

    confusion_path = (
        TABLE_DIR / f"{model_name}_top_confusions.csv"
    )

    with open(confusion_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "true_breed",
                "predicted_breed",
                "count"
            ]
        )

        writer.writeheader()
        writer.writerows(confusions)

    print(f"\nSaved: {output_path}")
    print(f"Saved: {confusion_path}")


def main():

    breed_names = load_breed_names()

    for model_name in MODELS:

        analyze_model(
            model_name,
            breed_names
        )

    print("\n" + "=" * 75)
    print("TEST ERROR ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
