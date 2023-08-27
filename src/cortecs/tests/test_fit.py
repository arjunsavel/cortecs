"""
tests all of the fitting functions.
"""
import unittest
from cortecs.fit.fit import *


class TestFitter(unittest.TestCase):
    """
    Test the fitter object itself
    """

    def test_available_methods(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        opac = ""  # not a real Opac object, just a test here

        self.assertRaises(ValueError, Fitter, opac, method="wrongmethod")
