import cv2
import numpy as np


GRAY_LEVELS = 32
DISTANCES = (1, 2)
ANGLES = (0, 45, 90, 135)

STATISTICS = (
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
    "entropy"
)


def quantize_image(image, levels=32):

    if image.ndim != 2:
        raise ValueError("Expected grayscale image")

    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    if not 2 <= levels <= 256:
        raise ValueError("levels must be between 2 and 256")

    quantized = (
        image.astype(np.uint16) * levels // 256
    )

    return quantized.astype(np.int32)


def compute_glcm(image, distance=1, angle=0, levels=32):

    if image.ndim != 2:
        raise ValueError("Expected a 2D quantized image")

    if distance < 1:
        raise ValueError("Distance must be positive")

    if angle not in (0, 45, 90, 135):
        raise ValueError("Unsupported angle")

    if image.min() < 0 or image.max() >= levels:
        raise ValueError("Pixel values outside quantized range")

    angle_rad = np.deg2rad(angle)

    dx = int(round(distance * np.cos(angle_rad)))
    dy = -int(round(distance * np.sin(angle_rad)))

    h, w = image.shape

    y0 = max(0, -dy)
    y1 = min(h, h - dy)

    x0 = max(0, -dx)
    x1 = min(w, w - dx)

    glcm = np.zeros((levels, levels), dtype=np.float64)

    if y0 >= y1 or x0 >= x1:
        return glcm

    source = image[y0:y1, x0:x1]

    target = image[
        y0 + dy:y1 + dy,
        x0 + dx:x1 + dx
    ]

    pairs = (
        source.ravel() * levels + target.ravel()
    )

    counts = np.bincount(
        pairs,
        minlength=levels * levels
    )

    glcm = counts.reshape(levels, levels).astype(np.float64)

    # Symmetric GLCM
    glcm = glcm + glcm.T

    total = glcm.sum()

    if total > 0:
        glcm /= total

    return glcm


def glcm_statistics(glcm):

    levels = glcm.shape[0]

    i, j = np.indices((levels, levels))

    difference = i - j

    contrast = np.sum(
        glcm * difference ** 2
    )

    homogeneity = np.sum(
        glcm / (1.0 + difference ** 2)
    )

    energy = np.sqrt(
        np.sum(glcm ** 2)
    )

    px = glcm.sum(axis=1)
    py = glcm.sum(axis=0)

    indices = np.arange(levels)

    mean_x = np.sum(indices * px)
    mean_y = np.sum(indices * py)

    std_x = np.sqrt(
        np.sum(((indices - mean_x) ** 2) * px)
    )

    std_y = np.sqrt(
        np.sum(((indices - mean_y) ** 2) * py)
    )

    if std_x < 1e-12 or std_y < 1e-12:
        correlation = 1.0
    else:
        correlation = np.sum(
            glcm
            * (i - mean_x)
            * (j - mean_y)
        ) / (std_x * std_y)

    positive = glcm[glcm > 0]

    entropy = -np.sum(
        positive * np.log2(positive)
    )

    return np.array(
        [
            contrast,
            homogeneity,
            energy,
            correlation,
            entropy
        ],
        dtype=np.float32
    )


def extract_glcm_features(
    image,
    levels=GRAY_LEVELS,
    distances=DISTANCES,
    angles=ANGLES
):

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    elif image.ndim == 2:

        gray = image

    else:
        raise ValueError("Invalid image shape")

    quantized = quantize_image(
        gray,
        levels=levels
    )

    features = []

    for distance in distances:

        for angle in angles:

            matrix = compute_glcm(
                quantized,
                distance=distance,
                angle=angle,
                levels=levels
            )

            stats = glcm_statistics(matrix)

            features.extend(stats)

    vector = np.asarray(
        features,
        dtype=np.float32
    )

    expected = len(distances) * len(angles) * 5

    if vector.shape != (expected,):
        raise RuntimeError("Unexpected feature count")

    if not np.all(np.isfinite(vector)):
        raise RuntimeError("Non-finite GLCM features")

    return vector


def get_glcm_feature_names(
    distances=DISTANCES,
    angles=ANGLES
):

    names = []

    for distance in distances:

        for angle in angles:

            for statistic in STATISTICS:

                names.append(
                    f"glcm_d{distance}_a{angle}_{statistic}"
                )

    return names
