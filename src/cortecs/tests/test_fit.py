"""
tests all of the fitting functions.
"""
import unittest
from cortecs.fit.fit import *
from cortecs.fit.fit_pca import *


class TestFitter(unittest.TestCase):
    """
    Test the fitter object itself
    """

    def test_unavailable_methods(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        opac = ""  # not a real Opac object, just a test here

        self.assertRaises(ValueError, Fitter, opac, method="wrongmethod")

    def test_available_methods_assigned_name(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        opac = ""  # not a real Opac object, just a test here

        fitter = Fitter(opac, method="pca")
        self.assertEqual(fitter.method, "pca")

    def test_available_methods_assigned_func(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        opac = ""  # not a real Opac object, just a test here

        fitter = Fitter(opac, method="pca")
        self.assertEqual(fitter.fit_func, fit_pca)
