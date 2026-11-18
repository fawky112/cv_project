import cv2
import numpy as np


IMAGE_SIZE = 224


def extract_sift_descriptors(image):

    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    elif image.ndim == 2:

        gray = image

    else:
        raise ValueError("Invalid image dimensions")

    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 image")

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None:

        descriptors = np.empty(
            (0, 128),
            dtype=np.float32
        )

    return keypoints, descriptors.astype(np.float32)
