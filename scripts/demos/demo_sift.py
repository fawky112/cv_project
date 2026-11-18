from pathlib import Path

import cv2

from src.features.sift import extract_sift_descriptors


DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

IMAGE_PATH = DATA_ROOT / "images" / "Abyssinian_100.jpg"

OUTPUT_ROOT = Path("results/figures")

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.resize(
    image,
    (224, 224),
    interpolation=cv2.INTER_AREA
)

keypoints, descriptors = extract_sift_descriptors(image)

print("=" * 60)
print("SIFT FEATURE EXTRACTION")
print("=" * 60)

print(f"Image Shape: {image.shape}")

print(f"Number of Keypoints: {len(keypoints)}")

print(f"Descriptors Shape: {descriptors.shape}")

print(f"Descriptor Type: {descriptors.dtype}")


visualization = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

output_path = OUTPUT_ROOT / "sift_keypoints.png"

cv2.imwrite(
    str(output_path),
    visualization
)

print(f"\nVisualization saved to: {output_path}")
