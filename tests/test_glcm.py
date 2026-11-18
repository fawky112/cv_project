import unittest

import numpy as np

from src.features.glcm import (
    quantize_image,
    compute_glcm,
    glcm_statistics,
    extract_glcm_features,
    get_glcm_feature_names
)


class TestGLCM(unittest.TestCase):

    def setUp(self):

        self.image = np.random.default_rng(42).integers(
            0,
            256,
            size=(128, 128),
            dtype=np.uint8
        )

    def test_quantization(self):

        image = np.array(
            [[0, 7, 8, 255]],
            dtype=np.uint8
        )

        result = quantize_image(image, levels=32)

        np.testing.assert_array_equal(
            result,
            [[0, 0, 1, 31]]
        )

    def test_glcm_shape(self):

        quantized = quantize_image(self.image)

        matrix = compute_glcm(
            quantized,
            distance=1,
            angle=0
        )

        self.assertEqual(
            matrix.shape,
            (32, 32)
        )

    def test_glcm_normalization(self):

        quantized = quantize_image(self.image)

        matrix = compute_glcm(quantized)

        self.assertAlmostEqual(
            matrix.sum(),
            1.0,
            places=6
        )

    def test_glcm_symmetry(self):

        quantized = quantize_image(self.image)

        matrix = compute_glcm(quantized)

        np.testing.assert_allclose(
            matrix,
            matrix.T
        )

    def test_known_glcm(self):

        image = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 0]
            ],
            dtype=np.int32
        )

        result = compute_glcm(
            image,
            distance=1,
            angle=0,
            levels=2
        )

        expected = np.array(
            [
                [2, 3],
                [3, 4]
            ],
            dtype=np.float64
        ) / 12.0

        np.testing.assert_allclose(
            result,
            expected
        )

    def test_feature_count(self):

        features = extract_glcm_features(
            self.image
        )

        self.assertEqual(
            features.shape,
            (40,)
        )

    def test_finite_features(self):

        features = extract_glcm_features(
            self.image
        )

        self.assertTrue(
            np.all(np.isfinite(features))
        )

    def test_constant_image(self):

        image = np.full(
            (128, 128),
            100,
            dtype=np.uint8
        )

        features = extract_glcm_features(image)

        self.assertTrue(
            np.all(np.isfinite(features))
        )

        # Constant image:
        # Contrast=0, Homogeneity=1, Energy=1,
        # Correlation=1, Entropy=0

        np.testing.assert_allclose(
            features[:5],
            [0, 1, 1, 1, 0],
            atol=1e-6
        )

    def test_feature_names(self):

        names = get_glcm_feature_names()

        self.assertEqual(len(names), 40)

        self.assertEqual(len(set(names)), 40)


if __name__ == "__main__":
    unittest.main()
