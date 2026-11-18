import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view


def mean_filter(img, kernel_size=3):

    assert kernel_size % 2 == 1

    pad = kernel_size // 2

    padded = np.pad(
        img.astype(np.float32),
        pad,
        mode="reflect"
    )

    windows = sliding_window_view(
        padded,
        (kernel_size, kernel_size)
    )

    result = windows.mean(axis=(-2, -1))

    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_filter(img, kernel_size=3, sigma=1.0):

    assert kernel_size % 2 == 1
    assert sigma > 0

    radius = kernel_size // 2

    y, x = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    kernel = np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    kernel = kernel / kernel.sum()

    padded = np.pad(
        img.astype(np.float32),
        radius,
        mode="reflect"
    )

    windows = sliding_window_view(
        padded,
        (kernel_size, kernel_size)
    )

    result = np.sum(
        windows * kernel,
        axis=(-2, -1)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(img, kernel_size=3):

    assert kernel_size % 2 == 1

    pad = kernel_size // 2

    padded = np.pad(
        img,
        pad,
        mode="reflect"
    )

    windows = sliding_window_view(
        padded,
        (kernel_size, kernel_size)
    )

    result = np.median(
        windows,
        axis=(-2, -1)
    )

    return result.astype(np.uint8)


def image_histogram(img):

    assert img.dtype == np.uint8

    histogram = np.zeros(256, dtype=np.int64)

    values, counts = np.unique(
        img,
        return_counts=True
    )

    histogram[values] = counts

    return histogram


def histogram_equalization(img):

    hist = image_histogram(img)

    cdf = np.cumsum(hist)

    nonzero = cdf[cdf > 0]

    if len(nonzero) == 0:
        return img.copy()

    cdf_min = nonzero[0]

    total = img.size

    if total == cdf_min:
        return img.copy()

    lut = np.round(
        (cdf - cdf_min) /
        (total - cdf_min) * 255
    )

    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[img]


def contrast_stretching(img):

    img = img.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val == min_val:
        return img.astype(np.uint8)

    result = (
        (img - min_val) /
        (max_val - min_val)
    ) * 255

    return np.clip(result, 0, 255).astype(np.uint8)


def intensity_normalization(img):

    img = img.astype(np.float32)

    mean = img.mean()
    std = img.std()

    normalized = (img - mean) / (std + 1e-8)

    return normalized


def resize_image(img, size):

    return cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_AREA
    )