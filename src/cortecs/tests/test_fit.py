"""
tests all of the fitting functions.
"""
import unittest
from cortecs.fit.fit import *
from cortecs.opac.opac import *
from cortecs.fit.fit_pca import *
import os
import sys

sys.path.insert(0, os.path.abspath("."))


class TestFitter(unittest.TestCase):
    """
    Test the fitter object itself
    """

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
    opac = Opac(cross_sec_filename, loader="platon", load_kwargs=load_kwargs)

    def test_unavailable_methods(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        opac = ""  # not a real Opac object, just a test here

        self.assertRaises(ValueError, Fitter, opac, method="wrongmethod")

    def test_available_methods_assigned_name(self):
        """
        If I pass a method that is sypported, an error should be raised.
        :return:
        """

        fitter = Fitter(self.opac, method="pca")
        self.assertEqual(fitter.method, "pca")

    def test_available_methods_assigned_func(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """

        fitter = Fitter(self.opac, method="pca")
        self.assertEqual(fitter.fit_func, fit_pca)
