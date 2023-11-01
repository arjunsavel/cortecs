"""
Test what happens when you link it all together.
"""
import unittest
from cortecs.fit.fit import *
from cortecs.opac.opac import *
from cortecs.fit.fit_pca import *
import os
import sys
import numpy as np

import cortecs
from cortecs.opac.opac import *
from cortecs.fit.fit import *
from cortecs.fit.fit_pca import *
from cortecs.eval.eval import *

sys.path.insert(0, os.path.abspath("."))


class TestIntegration(unittest.TestCase):
    def test_quickstart(self):
        """
        basically just the quickstart notebook. check that error < 10%.
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
        opac_obj = Opac(cross_sec_filename, loader="platon", load_kwargs=load_kwargs)
        fitter = Fitter(opac_obj, wav_ind=-2, nc=3)
        fitter.fit()

        evaluator = Evaluator(opac_obj, fitter)
        temperature = 300.0
        pressure = 100.0
        wavelength = 2.99401875e-05

        temperature_ind = np.where(np.isclose(opac_obj.T, temperature))[0][0]
        pressure_ind = np.where(np.isclose(opac_obj.P, pressure))[0][0]
        wavelength_ind = np.where(np.isclose(opac_obj.wl, wavelength))[0][0]

        AMU = 1.6605390666e-24  # atomic mass unit in cgs. From astropy!

        array_res = opac_obj.cross_section[temperature_ind][pressure_ind][
            wavelength_ind
        ]

        eval_res = np.log10(
            evaluator.eval(temperature, pressure, wavelength)
            * evaluator.load_obj.species_weight
            * AMU
            * 1e-4
        )
        percent_error = 100 * (eval_res - array_res) / array_res
        self.assertTrue(percent_error < 10)
