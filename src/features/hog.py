import cv2
import numpy as np


IMAGE_SIZE = 128
CELL_SIZE = 8
BLOCK_SIZE = 2
BLOCK_STRIDE = 1
N_BINS = 9

POOL_GRID = (4, 4)

EPS = 1e-6


def to_grayscale(image):

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    elif image.ndim == 2:
        gray = image

    else:
        raise ValueError("Expected grayscale or BGR image")

    return gray.astype(np.float32)


def compute_gradients(image):

    gray = to_grayscale(image)

    padded = np.pad(
        gray,
        ((1, 1), (1, 1)),
        mode="reflect"
    )

    gx = (
        padded[1:-1, 2:]
        - padded[1:-1, :-2]
    )

    gy = (
        padded[2:, 1:-1]
        - padded[:-2, 1:-1]
    )

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    orientation = (
        np.degrees(np.arctan2(gy, gx)) % 180.0
    )

    return magnitude, orientation


def compute_cell_histograms(
    magnitude,
    orientation,
    cell_size=CELL_SIZE,
    n_bins=N_BINS
):

    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation shapes differ")

    h, w = magnitude.shape

    if h % cell_size != 0 or w % cell_size != 0:
        raise ValueError(
            "Image dimensions must be divisible by cell size"
        )

    cells_y = h // cell_size
    cells_x = w // cell_size

    bin_width = 180.0 / n_bins

    bin_position = orientation / bin_width

    lower_bin = np.floor(
        bin_position
    ).astype(np.int32) % n_bins

    upper_bin = (lower_bin + 1) % n_bins

    upper_weight = bin_position - np.floor(bin_position)

    lower_weight = 1.0 - upper_weight

    yy, xx = np.indices((h, w))

    cell_y = yy // cell_size
    cell_x = xx // cell_size

    cell_index = cell_y * cells_x + cell_x

    histogram_size = cells_y * cells_x * n_bins

    lower_index = (
        cell_index * n_bins + lower_bin
    ).ravel()

    upper_index = (
        cell_index * n_bins + upper_bin
    ).ravel()

    lower_votes = (
        magnitude * lower_weight
    ).ravel()

    upper_votes = (
        magnitude * upper_weight
    ).ravel()

    histograms = np.bincount(
        lower_index,
        weights=lower_votes,
        minlength=histogram_size
    )

    histograms += np.bincount(
        upper_index,
        weights=upper_votes,
        minlength=histogram_size
    )

    return histograms.reshape(
        cells_y,
        cells_x,
        n_bins
    ).astype(np.float32)


def l2_hys_normalize(
    vector,
    eps=EPS,
    clip=0.2
):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.sqrt(
        np.sum(vector ** 2) + eps ** 2
    )

    vector = vector / norm

    vector = np.minimum(vector, clip)

    norm = np.sqrt(
        np.sum(vector ** 2) + eps ** 2
    )

    return vector / norm


def compute_hog_blocks(
    cell_histograms,
    block_size=BLOCK_SIZE,
    block_stride=BLOCK_STRIDE
):

    cells_y, cells_x, n_bins = cell_histograms.shape

    if block_size > min(cells_y, cells_x):
        raise ValueError("Block size exceeds cell grid")

    if block_stride < 1:
        raise ValueError("Block stride must be positive")

    features = []

    for y in range(
        0,
        cells_y - block_size + 1,
        block_stride
    ):

        for x in range(
            0,
            cells_x - block_size + 1,
            block_stride
        ):

            block = cell_histograms[
                y:y + block_size,
                x:x + block_size
            ]

            block_vector = block.ravel()

            normalized = l2_hys_normalize(
                block_vector
            )

            features.extend(normalized)

    return np.asarray(
        features,
        dtype=np.float32
    )


def compute_pooled_hog(
    cell_histograms,
    pool_grid=POOL_GRID
):

    cells_y, cells_x, n_bins = cell_histograms.shape

    pool_y, pool_x = pool_grid

    if pool_y < 1 or pool_x < 1:
        raise ValueError("Invalid pooling grid")

    if cells_y % pool_y != 0 or cells_x % pool_x != 0:
        raise ValueError(
            "Cell grid must be divisible by pooling grid"
        )

    group_y = cells_y // pool_y
    group_x = cells_x // pool_x

    reshaped = cell_histograms.reshape(
        pool_y,
        group_y,
        pool_x,
        group_x,
        n_bins
    )

    pooled = reshaped.mean(
        axis=(1, 3)
    )

    # Normalize each pooled orientation histogram.
    norms = np.sqrt(
        np.sum(pooled ** 2, axis=2, keepdims=True)
        + EPS ** 2
    )

    pooled = pooled / norms

    return pooled.ravel().astype(np.float32)


def extract_hog_features(
    image,
    cell_size=CELL_SIZE,
    block_size=BLOCK_SIZE,
    n_bins=N_BINS
):

    magnitude, orientation = compute_gradients(
        image
    )

    cell_histograms = compute_cell_histograms(
        magnitude,
        orientation,
        cell_size=cell_size,
        n_bins=n_bins
    )

    features = compute_hog_blocks(
        cell_histograms,
        block_size=block_size
    )

    if not np.all(np.isfinite(features)):
        raise RuntimeError("Non-finite HOG features")

    return features


def extract_pooled_hog_features(
    image,
    cell_size=CELL_SIZE,
    n_bins=N_BINS,
    pool_grid=POOL_GRID
):

    magnitude, orientation = compute_gradients(
        image
    )

    cell_histograms = compute_cell_histograms(
        magnitude,
        orientation,
        cell_size=cell_size,
        n_bins=n_bins
    )

    features = compute_pooled_hog(
        cell_histograms,
        pool_grid=pool_grid
    )

    if not np.all(np.isfinite(features)):
        raise RuntimeError("Non-finite pooled HOG features")

    return features


def extract_both_hog_features(image):

    magnitude, orientation = compute_gradients(
        image
    )

    cells = compute_cell_histograms(
        magnitude,
        orientation
    )

    hog = compute_hog_blocks(cells)

    pooled_hog = compute_pooled_hog(cells)

    return hog, pooled_hog
