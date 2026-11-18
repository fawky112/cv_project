import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from preprocessing import (
    mean_filter,
    gaussian_filter,
    median_filter,
    histogram_equalization,
    contrast_stretching,
    intensity_normalization,
    resize_image,
    image_histogram
)


DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

image_path = DATA_ROOT / "images" / "Abyssinian_100.jpg"

img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(image_path)

img = resize_image(img, 128)

normalized = intensity_normalization(img)

results = {
    "Original": img,
    "Mean Filter": mean_filter(img),
    "Gaussian Filter": gaussian_filter(img),
    "Median Filter": median_filter(img),
    "Histogram Equalization": histogram_equalization(img),
    "Contrast Stretching": contrast_stretching(img),
    "Intensity Normalization": normalized,
    "Resize 96x96": resize_image(img, 96),
    "Resize 64x64": resize_image(img, 64)
}


output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(
    len(results),
    2,
    figsize=(12, 24)
)

for i, (name, result) in enumerate(results.items()):

    axes[i, 0].imshow(
        result,
        cmap="gray",
        vmin=0 if name != "Intensity Normalization" else None,
        vmax=255 if name != "Intensity Normalization" else None
    )

    axes[i, 0].set_title(name)
    axes[i, 0].axis("off")

    if name == "Intensity Normalization":

        hist, edges = np.histogram(
            result,
            bins=50
        )

        axes[i, 1].stairs(hist, edges)

    else:

        hist = image_histogram(result)

        axes[i, 1].plot(
            np.arange(256),
            hist
        )

    axes[i, 1].set_title(f"{name} Histogram")
    axes[i, 1].set_xlabel("Pixel Intensity")
    axes[i, 1].set_ylabel("Frequency")


plt.tight_layout()

save_path = output_dir / "preprocessing_comparison.png"

plt.savefig(
    save_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Preprocessing completed successfully!")

print(f"Figure saved to: {save_path}")