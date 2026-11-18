import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

GROUPS = {
    "extraction": [
        "extract_color_dataset.py",
        "extract_lbp_dataset.py",
        "extract_glcm_dataset.py",
        "extract_hog_dataset.py",
        "extract_sift_dataset.py",
        "extract_transformer.py",
        "extract_transformer_test.py",
        "extract_remaining_test.py",
        "build_bovw.py",
        "combine_handcrafted.py",
        "combine_all_features.py",
        "select_mrmr.py",
        "prepare_mrmr_test.py",
    ],

    "evaluation": [
        "evaluate_color.py",
        "evaluate_lbp.py",
        "evaluate_glcm.py",
        "evaluate_hog.py",
        "evaluate_handcrafted.py",
        "evaluate_bovw.py",
        "evaluate_transformer.py",
        "evaluate_mrmr.py",
        "evaluate_all_features.py",
        "evaluate_final_test.py",
        "evaluate_remaining_test.py",
    ],

    "experiments": [
        "grid_search_color.py",
        "preprocessing_experiments.py",
    ],

    "analysis": [
        "analyze_mrmr.py",
        "analyze_test_errors.py",
        "analyze_computational_cost.py",
        "verify_feature_pipeline.py",
        "audit_project.py",
    ],

    "visualization": [
        "generate_final_figures.py",
        "visualize_test_errors.py",
    ],
}


def fix_project_paths(code):

    old_path = "Path(__file__).resolve().parent"

    new_path = "Path(__file__).resolve().parents[2]"

    return code.replace(old_path, new_path)


def organize():

    print("=" * 70)
    print("ORGANIZING CSE480 PROJECT")
    print("=" * 70)

    moved = 0
    missing = []

    for group, filenames in GROUPS.items():

        destination = ROOT / "scripts" / group

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        (destination / "__init__.py").touch(
            exist_ok=True
        )

        for filename in filenames:

            source = ROOT / filename
            target = destination / filename

            if target.exists():
                print(f"ALREADY EXISTS: {target.relative_to(ROOT)}")
                continue

            if not source.exists():
                missing.append(filename)
                continue

            code = source.read_text(
                encoding="utf-8"
            )

            code = fix_project_paths(code)

            target.write_text(
                code,
                encoding="utf-8"
            )

            source.unlink()

            moved += 1

            print(
                f"MOVED: {filename} "
                f"-> scripts/{group}/"
            )

    (ROOT / "scripts" / "__init__.py").touch(
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(f"Files moved: {moved}")

    if missing:

        print("\nFiles not found at project root:")

        for filename in missing:
            print(f"  {filename}")

    print("\nPROJECT ORGANIZATION COMPLETED")


if __name__ == "__main__":
    organize()
