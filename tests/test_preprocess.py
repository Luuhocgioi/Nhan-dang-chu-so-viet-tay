import unittest
import cv2
import numpy as np
from src.preprocess import preprocess, EmptyImageError, read_image


class PreprocessTests(unittest.TestCase):
    def digit(self):
        image = np.zeros((160, 200), np.uint8)
        cv2.line(image, (100, 30), (100, 130), 255, 12)
        return image

    def test_blank_both_polarities(self):
        for value in (0, 255):
            with self.assertRaises(EmptyImageError):
                preprocess(np.full((100, 100), value, np.uint8))

    def test_polarity_invariance_and_tensor_contract(self):
        a = preprocess(self.digit())
        b = preprocess(255-self.digit())
        np.testing.assert_allclose(a, b)
        self.assertEqual(a.shape, (1, 28, 28, 1))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(0 <= a.min() < a.max() <= 1)

    def test_translation_invariance(self):
        shifted = cv2.warpAffine(self.digit(), np.float32([[1, 0, 30], [0, 1, 10]]), (200, 160))
        np.testing.assert_allclose(preprocess(self.digit()), preprocess(shifted))

    def test_alpha_black_ink_on_transparent_background(self):
        alpha = self.digit()
        bgra = np.zeros((*alpha.shape, 4), np.uint8)
        bgra[..., 3] = alpha
        np.testing.assert_allclose(preprocess(bgra), preprocess(alpha))

    def test_isolated_noise_is_empty(self):
        image = np.zeros((200, 200), np.uint8)
        image[50, 50] = 255
        with self.assertRaises(EmptyImageError):
            preprocess(image)

    def test_bad_file(self):
        with self.assertRaises(ValueError):
            read_image('requirements.txt')


if __name__ == '__main__':
    unittest.main()
