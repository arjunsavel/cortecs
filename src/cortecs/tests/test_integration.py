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
import random
import tensorflow as tf

import cortecs
from cortecs.opac.opac import *
from cortecs.fit.fit import *
from cortecs.fit.fit_pca import *
from cortecs.eval.eval import *
from cortecs.opt.opt import *
import pickle

seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)

sys.path.insert(0, os.path.abspath("."))


class TestIntegration(unittest.TestCase):
    T_filename = os.path.abspath(".") + "/src/cortecs/tests/temperatures.npy"
    P_filename = os.path.abspath(".") + "/src/cortecs/tests/pressures.npy"
    wl_filename = os.path.abspath(".") + "/src/cortecs/tests/wavelengths.npy"
    cross_sec_filename = (
        os.path.abspath(".") + "/src/cortecs/tests/absorb_coeffs_C2H4.npy"
    )

    def test_quickstart(self):
        """
        basically just the quickstart notebook. check that error < 10%.
        """

        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        opac_obj = Opac(
            self.cross_sec_filename, loader="platon", load_kwargs=load_kwargs
        )
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

    def test_nn(self):
        """
        basically just the neural net notebook. check that error < 10%.
        """
        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        opac_obj = Opac(
            self.cross_sec_filename, loader="platon", load_kwargs=load_kwargs
        )
        fitter = Fitter(opac_obj, method="neural_net")
        res = cortecs.fit.fit_neural_net.fit_neural_net(
            fitter.opac.cross_section[:, :, -2],
            fitter.opac.T,
            fitter.opac.P,
            None,
            all_wl=False,
            n_layers=3,
            n_neurons=8,
            activation="sigmoid",
            learn_rate=0.04,
            loss="mean_squared_error",
            epochs=4000,
            verbose=0,
            sequential_model=None,
            plot=False,
        )
        history, neural_network = res
        P_unraveled = unravel_data(fitter.opac.P, fitter.opac.T, None, tileboth=True)
        T_unraveled = unravel_data(fitter.opac.T, fitter.opac.P, None, tileboth=False)
        input_array = np.column_stack([T_unraveled, P_unraveled])

        npres = len(fitter.opac.P)
        ntemp = len(fitter.opac.T)

        predictions = neural_network.predict(input_array)
        percent_errors = (
            100
            * (predictions.reshape(ntemp, npres) - fitter.opac.cross_section[:, :, -1])
            / predictions.reshape(ntemp, npres)
        )
        median_err = np.median(np.abs(percent_errors))
        save_neural_network(neural_network, "test_nn.pkl")
        with open("test_nn.pkl", "rb") as f:
            all_weights, all_biases = pickle.load(f)

        n_layers = len(all_weights)
        res1 = eval_neural_net(100, 1e-4, n_layers, all_weights, all_biases)
        res2 = predictions[0]
        self.assertTrue(np.isclose(res1, res2) and median_err < 10)

    def test_polynomial(self):
        """
        basically from the tutorial. just check that it runs and that it's close to the expected value.
        :return:
        """
        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        opac_obj = Opac(
            self.cross_sec_filename, loader="platon", load_kwargs=load_kwargs
        )
        fitter = Fitter(opac_obj, method="polynomial")
        fitter.fit()
        evaluator = Evaluator(opac_obj, fitter)
        temperature = 300.0
        pressure = 100
        wavelength = 2.99401875e-05

        res = evaluator.eval(pressure, temperature, wavelength)
        self.assertTrue(np.isclose(res, 1.3991368e-05, atol=1e-8))

    def test_optimize(self):
        """
        just the tutorial. once more!
        :return:
        """
        # reset random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        opac_obj = Opac(
            self.cross_sec_filename, loader="platon", load_kwargs=load_kwargs
        )
        fitter = Fitter(opac_obj, method="neural_net")
        res = cortecs.fit.fit_neural_net.fit_neural_net(
            fitter.opac.cross_section[:, :, -2],
            fitter.opac.T,
            fitter.opac.P,
            None,
            all_wl=False,
            n_layers=3,
            n_neurons=8,
            activation="sigmoid",
            learn_rate=0.04,
            loss="mean_squared_error",
            epochs=4000,
            verbose=0,
            sequential_model=None,
            plot=False,
        )
        optimizer = Optimizer(fitter)
        max_size = 1.6
        max_evaluations = 8
        optimizer.optimize(max_size, max_evaluations)
        self.assertTrue(
            optimizer.best_params
            == {
                "n_layers": 2,
                "n_neurons": 2.0,
                "activation": "relu",
                "learn_rate": 0.01,
            }
        )
