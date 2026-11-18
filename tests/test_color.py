import unittest

import cv2
import numpy as np

from src.features.color import (
    extract_color_features,
    get_color_feature_names,
    normalized_histogram,
    channel_statistics
)


class TestColorFeatures(unittest.TestCase):

    def setUp(self):

        self.image = np.random.default_rng(42).integers(
            0,
            256,
            size=(128, 128, 3),
            dtype=np.uint8
        )

    def test_feature_count(self):

        features = extract_color_features(self.image)

        self.assertEqual(
            features.shape,
            (88,)
        )

    def test_feature_dtype(self):

        features = extract_color_features(self.image)

        self.assertEqual(
            features.dtype,
            np.float32
        )

    def test_finite_values(self):

        features = extract_color_features(self.image)

        self.assertTrue(
            np.all(np.isfinite(features))
        )

    def test_constant_image(self):

        image = np.full(
            (128, 128, 3),
            100,
            dtype=np.uint8
        )

        features = extract_color_features(image)

        self.assertTrue(
            np.all(np.isfinite(features))
        )

    def test_histogram_normalization(self):

        channel = self.image[:, :, 0]

        hist = normalized_histogram(
            channel,
            bins=16,
            value_range=(0, 256)
        )

        self.assertAlmostEqual(
            float(hist.sum()),
            1.0,
            places=5
        )

    def test_constant_channel_statistics(self):

        channel = np.full(
            (10, 10),
            50,
            dtype=np.uint8
        )

        stats = channel_statistics(channel)

        np.testing.assert_allclose(
            stats,
            [50.0, 0.0, 0.0, 0.0]
        )

    def test_feature_names(self):

        names = get_color_feature_names()

        self.assertEqual(len(names), 88)

        self.assertEqual(len(set(names)), 88)


if __name__ == "__main__":
    unittest.main()
