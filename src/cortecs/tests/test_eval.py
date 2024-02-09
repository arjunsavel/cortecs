"""
tests all of the evaluation functions.
"""

import unittest
import numpy as np
import os
from cortecs.eval.eval_pca import *
from cortecs.fit.fit import *
from cortecs.opac.opac import *


class TestEvalPca(unittest.TestCase):
    T_filename = os.path.abspath(".") + "/src/cortecs/tests/temperatures.npy"
    P_filename = os.path.abspath(".") + "/src/cortecs/tests/pressures.npy"
    wl_filename = os.path.abspath(".") + "/src/cortecs/tests/wavelengths.npy"
    cross_sec_filename = (
        os.path.abspath(".") + "/src/cortecs/tests/absorb_coeffs_C2H4.npy"
    )
    load_kwargs = {
        "T_filename": T_filename,
        "P_filename": P_filename,
        "wl_filename": wl_filename,
    }
    opac_obj = Opac(cross_sec_filename, loader="platon", load_kwargs=load_kwargs)
    fitter = Fitter(opac_obj, wav_ind=-2, nc=3)
    fitter.fit()
    evaluator = Evaluator(opac_obj, fitter)

    def test_known_value(self):
        known_value = -48.730205813777914
        test_pca_coeffs = np.load("src/cortecs/tests/test_pca_coeffs.npy")
        test_pca_vectors = np.load("src/cortecs/tests/test_pca_vectors.npy")

        test_val = eval_pca_ind_wav(0, 0, test_pca_vectors, test_pca_coeffs)
        np.testing.assert_almost_equal(test_val, known_value, decimal=5)

    def test_eval_OOB_temperature_breaks(self):
        """
        if I try to evaluate at a temperature that's out of bounds, it should break.
        """
        # use the fitter I defined in this object to break things
        temperature = 1e10
        pressure = 2
        wavelength = 2.99401875e-05
        with self.assertRaises(ValueError):
            self.evaluator.eval(temperature, pressure, wavelength)

    def test_eval_OOB_pressure_breaks(self):
        """
        if I try to evaluate at a pressure that's out of bounds, it should break.
        """
        # use the fitter I defined in this object to break things
        temperature = 300
        pressure = 1e10
        wavelength = 2.99401875e-05
        with self.assertRaises(ValueError):
            self.evaluator.eval(temperature, pressure, wavelength)

    def test_eval_OOB_wavelength_breaks(self):
        """
        if I try to evaluate at a wavelength that's out of bounds, it should break.
        """
        # use the fitter I defined in this object to break things
        temperature = 300
        pressure = 2
        wavelength = 1e10
        with self.assertRaises(ValueError):
            self.evaluator.eval(temperature, pressure, wavelength)
