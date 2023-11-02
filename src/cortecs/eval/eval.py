"""
The high-level API for evaluating. Requires the Opac object and the Fitter object.
"""
from cortecs.eval.eval_neural_net import *
from cortecs.eval.eval_pca import *
from cortecs.eval.eval_polynomial import *
from cortecs.opac.io import *

AMU = 1.6605390666e-24  # atomic mass unit in cgs. From astropy!


class Evaluator(object):
    """
    evaluates the opacity data.
    """

    method_dict = {
        "pca": eval_pca,
        "neural_net": eval_neural_net,
        "polynomial": eval_polynomial,
    }

    def __init__(self, opac, fitter, **eval_kwargs):
        """ """
        method = fitter.method
        self.opac = opac
        self.load_obj = opac.load_obj
        self.wl = opac.wl
        self.P = opac.P
        self.T = opac.T

        if not hasattr(fitter, "fitter_results"):
            raise ValueError("Fitter must be run before being passed to an Evaluator.")

        self.fitter_results = fitter.fitter_results
        self.eval_kwargs = eval_kwargs

        self.eval_func = self.method_dict[method]

        return

    def eval(self, temperature, pressure, wavelength):
        """
        evaluates the opacity data.
        """
        eval_result = self.eval_func(
            temperature,
            pressure,
            wavelength,
            self.T,
            self.P,
            self.wl,
            self.fitter_results,
            **self.eval_kwargs
        )
        # we need to recast the units if PLATON was used.
        if type(self.load_obj) == loader_platon:
            eval_result = 10**eval_result
            eval_result = eval_result / self.load_obj.species_weight / AMU / 1e-4

        return eval_result
