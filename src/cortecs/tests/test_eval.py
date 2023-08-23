"""
tests all of the evaluation functions.
"""

import unittest
import numpy as np

from cortecs.eval.eval_pca import *


class TestEvalPca(unittest.TestCase):
    def test_known_value(self):
        known_value = -48.730205813777914
        test_pca_coeffs = np.load("src/cortecs/tests/test_pca_coeffs.npy")
        test_pca_vectors = np.load("src/cortecs/tests/test_pca_vectors.npy")

        test_val = eval_pca(0, 0, test_pca_vectors, test_pca_coeffs, n_components=3)
        np.testing.assert_almost_equal(test_val, known_value, decimal=5)
