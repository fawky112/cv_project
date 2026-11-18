import cv2
import numpy as np


RGB_BINS = 16
HUE_BINS = 16


def channel_statistics(channel):

    x = channel.astype(np.float64).ravel()

    mean = np.mean(x)

    std = np.std(x)

    if std < 1e-10:
        skewness = 0.0
        kurtosis = 0.0

    else:
        z = (x - mean) / std

        skewness = np.mean(z ** 3)

        kurtosis = np.mean(z ** 4) - 3.0

    return np.array(
        [mean, std, skewness, kurtosis],
        dtype=np.float32
    )


def normalized_histogram(channel, bins, value_range):

    hist, _ = np.histogram(
        channel,
        bins=bins,
        range=value_range
    )

    hist = hist.astype(np.float32)

    total = np.sum(hist)

    if total > 0:
        hist = hist / total

    return hist


def extract_color_features(image):

    if image is None:
        raise ValueError("Input image is None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image")

    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    features = []

    # RGB statistical features
    for channel_idx in range(3):

        stats = channel_statistics(
            rgb[:, :, channel_idx]
        )

        features.extend(stats)

    # HSV statistical features
    for channel_idx in range(3):

        stats = channel_statistics(
            hsv[:, :, channel_idx]
        )

        features.extend(stats)

    # RGB histograms
    for channel_idx in range(3):

        hist = normalized_histogram(
            rgb[:, :, channel_idx],
            bins=RGB_BINS,
            value_range=(0, 256)
        )

        features.extend(hist)

    # Hue histogram
    hue_hist = normalized_histogram(
        hsv[:, :, 0],
        bins=HUE_BINS,
        value_range=(0, 180)
    )

    features.extend(hue_hist)

    feature_vector = np.asarray(
        features,
        dtype=np.float32
    )

    if feature_vector.shape != (88,):
        raise RuntimeError(
            f"Expected 88 features, got {feature_vector.shape}"
        )

    if not np.all(np.isfinite(feature_vector)):
        raise RuntimeError("Non-finite feature values detected")

    return feature_vector


def get_color_feature_names():

    names = []

    statistics = [
        "mean",
        "std",
        "skewness",
        "kurtosis"
    ]

    for space, channels in [
        ("rgb", ["r", "g", "b"]),
        ("hsv", ["h", "s", "v"])
    ]:

        for channel in channels:

            for stat in statistics:

                names.append(
                    f"{space}_{channel}_{stat}"
                )

    for channel in ["r", "g", "b"]:

        for bin_idx in range(RGB_BINS):

            names.append(
                f"rgb_{channel}_hist_bin_{bin_idx}"
            )

    for bin_idx in range(HUE_BINS):

        names.append(
            f"hue_hist_bin_{bin_idx}"
        )

    assert len(names) == 88

    return names