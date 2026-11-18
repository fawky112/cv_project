import time

from pathlib import Path

import cv2
import numpy as np

from src.features.color import (
    extract_color_features,
    get_color_feature_names
)


DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

IMAGE_PATH = DATA_ROOT / "images" / "Abyssinian_100.jpg"


image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)


image = cv2.resize(
    image,
    (128, 128),
    interpolation=cv2.INTER_AREA
)


start = time.perf_counter()

features = extract_color_features(image)

elapsed = time.perf_counter() - start


names = get_color_feature_names()


print("=" * 60)
print("COLOR FEATURE EXTRACTION")
print("=" * 60)

print(f"Image: {IMAGE_PATH.name}")

print(f"Image Shape: {image.shape}")

print(f"Feature Vector Shape: {features.shape}")

print(f"Feature Vector Type: {features.dtype}")

print(f"Extraction Time: {elapsed * 1000:.3f} ms")

print(f"NaN Values: {np.isnan(features).sum()}")

print(f"Inf Values: {np.isinf(features).sum()}")

print("\nFIRST 24 FEATURES")
print("-" * 60)

for name, value in zip(names[:24], features[:24]):

    print(f"{name:25s}: {value:.6f}")


print("\nHISTOGRAM CHECK")

rgb_histograms = features[24:72]

hue_histogram = features[72:88]

for i, channel in enumerate(["R", "G", "B"]):

    hist = rgb_histograms[i * 16:(i + 1) * 16]

    print(f"{channel} Histogram Sum: {hist.sum():.6f}")

print(f"Hue Histogram Sum: {hue_histogram.sum():.6f}")
