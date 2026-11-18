import unittest

import numpy as np

from src.features.hog import (
    compute_gradients,
    compute_cell_histograms,
    l2_hys_normalize,
    extract_hog_features,
    extract_pooled_hog_features,
    extract_both_hog_features
)


class TestHOG(unittest.TestCase):

    def setUp(self):

        self.image = np.random.default_rng(42).integers(
            0,
            256,
            size=(128, 128),
            dtype=np.uint8
        )

    def test_gradient_shape(self):

        magnitude, orientation = compute_gradients(
            self.image
        )

        self.assertEqual(
            magnitude.shape,
            (128, 128)
        )

        self.assertEqual(
            orientation.shape,
            (128, 128)
        )

    def test_orientation_range(self):

        _, orientation = compute_gradients(self.image)

        self.assertTrue(
            np.all(orientation >= 0)
        )

        self.assertTrue(
            np.all(orientation < 180)
        )

    def test_horizontal_gradient(self):

        image = np.tile(
            np.arange(128, dtype=np.uint8),
            (128, 1)
        )

        magnitude, orientation = compute_gradients(
            image
        )

        np.testing.assert_allclose(
            orientation[1:-1, 1:-1],
            0,
            atol=1e-5
        )

        self.assertTrue(
            np.all(magnitude[1:-1, 1:-1] > 0)
        )

    def test_cell_histogram_shape(self):

        magnitude, orientation = compute_gradients(
            self.image
        )

        histograms = compute_cell_histograms(
            magnitude,
            orientation
        )

        self.assertEqual(
            histograms.shape,
            (16, 16, 9)
        )

    def test_histogram_vote_conservation(self):

        magnitude, orientation = compute_gradients(
            self.image
        )

        histograms = compute_cell_histograms(
            magnitude,
            orientation
        )

        self.assertAlmostEqual(
            float(histograms.sum()),
            float(magnitude.sum()),
            delta=float(magnitude.sum()) * 1e-5
        )

    def test_hog_feature_count(self):

        features = extract_hog_features(self.image)

        self.assertEqual(
            features.shape,
            (8100,)
        )

    def test_pooled_feature_count(self):

        features = extract_pooled_hog_features(
            self.image
        )

        self.assertEqual(
            features.shape,
            (144,)
        )

    def test_finite_features(self):

        hog, pooled = extract_both_hog_features(
            self.image
        )

        self.assertTrue(
            np.all(np.isfinite(hog))
        )

        self.assertTrue(
            np.all(np.isfinite(pooled))
        )

    def test_constant_image(self):

        image = np.full(
            (128, 128),
            100,
            dtype=np.uint8
        )

        hog, pooled = extract_both_hog_features(
            image
        )

        np.testing.assert_allclose(
            hog,
            0,
            atol=1e-6
        )

        np.testing.assert_allclose(
            pooled,
            0,
            atol=1e-6
        )

    def test_l2_hys_normalization(self):

        vector = np.array(
            [3.0, 4.0, 5.0, 6.0],
            dtype=np.float32
        )

        normalized = l2_hys_normalize(vector)

        self.assertAlmostEqual(
            float(np.linalg.norm(normalized)),
            1.0,
            places=5
        )


if __name__ == "__main__":
    unittest.main()
