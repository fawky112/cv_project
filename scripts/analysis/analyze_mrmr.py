import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SELECTED_DIR = ROOT / "features" / "selected"

RESULT_DIR = ROOT / "results" / "tables"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_selection(k):

    path = SELECTED_DIR / f"mrmr_{k}_dictionary.json"

    with open(path, "r") as file:
        features = json.load(file)

    counts = Counter(
        feature["feature_group"]
        for feature in features
    )

    print("\n" + "=" * 65)
    print(f"mRMR TOP {k} FEATURE DISTRIBUTION")
    print("=" * 65)

    for family, count in counts.most_common():

        percentage = count / k * 100

        print(
            f"{family:15s} | "
            f"{count:3d} Features | "
            f"{percentage:6.2f}%"
        )

    return features, counts


def main():

    summary = []

    for k in [20, 50, 100]:

        features, counts = analyze_selection(k)

        for family, count in counts.items():

            summary.append({
                "selected_k": k,
                "feature_family": family,
                "feature_count": count,
                "percentage": count / k * 100
            })

    output_path = (
        RESULT_DIR / "mrmr_feature_distribution.csv"
    )

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "selected_k",
                "feature_family",
                "feature_count",
                "percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 65)
    print("TOP 20 SELECTED FEATURES")
    print("=" * 65)

    with open(
        SELECTED_DIR / "mrmr_100_dictionary.json",
        "r"
    ) as file:

        selected = json.load(file)

    for rank, feature in enumerate(selected[:20], start=1):

        print(
            f"{rank:3d} | "
            f"{feature['feature_group']:15s} | "
            f"{feature['name']}"
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
