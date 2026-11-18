import unittest
import numpy as np

from src.features.lbp import (
    build_uniform_lbp_lut,
    compute_lbp,
    extract_lbp_features
)


class TestLBP(unittest.TestCase):

    def setUp(self):
        self.image = np.random.default_rng(42).integers(
            0, 256,
            size=(128, 128),
            dtype=np.uint8
        )

    def test_uniform_patterns(self):

        lut = build_uniform_lbp_lut()

        self.assertEqual(
            np.sum(lut != 58),
            58
        )

    def test_lbp_image_shape(self):

        lbp = compute_lbp(self.image)

        self.assertEqual(
            lbp.shape,
            self.image.shape
        )

    def test_feature_count(self):

        features = extract_lbp_features(self.image)

        self.assertEqual(
            features.shape,
            (1003,)
        )

    def test_finite_values(self):

        features = extract_lbp_features(self.image)

        self.assertTrue(
            np.all(np.isfinite(features))
        )

    def test_histogram_normalization(self):

        features = extract_lbp_features(self.image)

        histograms = features.reshape(17, 59)

        np.testing.assert_allclose(
            histograms.sum(axis=1),
            np.ones(17),
            atol=1e-5
        )

    def test_constant_image(self):

        image = np.full(
            (128, 128),
            100,
            dtype=np.uint8
        )

        features = extract_lbp_features(image)

        self.assertTrue(
            np.all(np.isfinite(features))
        )


if __name__ == "__main__":
    unittest.main()
