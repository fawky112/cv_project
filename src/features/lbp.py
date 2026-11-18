import cv2
import numpy as np


def build_uniform_lbp_lut(P=8):

    if P != 8:
        raise ValueError("This implementation supports P=8")

    lut = np.zeros(256, dtype=np.uint8)

    uniform_index = 0

    for pattern in range(256):

        bits = np.array(
            [(pattern >> i) & 1 for i in range(P)],
            dtype=np.uint8
        )

        transitions = np.sum(
            bits != np.roll(bits, 1)
        )

        if transitions <= 2:
            lut[pattern] = uniform_index
            uniform_index += 1
        else:
            lut[pattern] = 58

    assert uniform_index == 58

    return lut


def compute_lbp(image, radius=1, neighbours=8):

    if image.ndim != 2:
        raise ValueError("LBP requires grayscale image")

    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    if radius <= 0:
        raise ValueError("Radius must be positive")

    if neighbours != 8:
        raise ValueError("Currently supports 8 neighbours")

    h, w = image.shape

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    center = image.astype(np.float32)

    codes = np.zeros((h, w), dtype=np.uint8)

    for p in range(neighbours):

        angle = 2 * np.pi * p / neighbours

        dx = radius * np.cos(angle)
        dy = -radius * np.sin(angle)

        map_x = xx + dx
        map_y = yy + dy

        neighbour = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        bit = (
            neighbour.astype(np.float32) >= center
        ).astype(np.uint8)

        codes |= bit << p

    return codes


def normalized_lbp_histogram(lbp_image, bins=59):

    hist = np.bincount(
        lbp_image.ravel(),
        minlength=bins
    ).astype(np.float32)

    hist /= max(float(hist.sum()), 1.0)

    return hist


def extract_lbp_features(
    image,
    radius=1,
    neighbours=8,
    grid=(4, 4)
):

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    codes = compute_lbp(
        gray,
        radius=radius,
        neighbours=neighbours
    )

    lut = build_uniform_lbp_lut(neighbours)

    uniform_lbp = lut[codes]

    features = []

    features.extend(
        normalized_lbp_histogram(uniform_lbp)
    )

    rows, cols = grid

    if rows <= 0 or cols <= 0:
        raise ValueError("Grid dimensions must be positive")

    if rows > gray.shape[0] or cols > gray.shape[1]:
        raise ValueError("Grid exceeds image dimensions")

    row_edges = np.linspace(
        0, gray.shape[0], rows + 1, dtype=int
    )

    col_edges = np.linspace(
        0, gray.shape[1], cols + 1, dtype=int
    )

    for r in range(rows):
        for c in range(cols):

            cell = uniform_lbp[
                row_edges[r]:row_edges[r + 1],
                col_edges[c]:col_edges[c + 1]
            ]

            hist = normalized_lbp_histogram(cell)

            features.extend(hist)

    vector = np.asarray(
        features,
        dtype=np.float32
    )

    expected = 59 * (1 + rows * cols)

    assert vector.shape == (expected,)
    assert np.all(np.isfinite(vector))

    return vector
