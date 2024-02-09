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
        self.fit_kwargs = fitter.fitter_kwargs

        if not hasattr(fitter, "fitter_results"):
            raise ValueError("Fitter must be run before being passed to an Evaluator.")

        self.fitter_results = fitter.fitter_results
        self.eval_kwargs = eval_kwargs

        self.eval_func = self.method_dict[method]

        return

    def eval(self, temperature, pressure, wavelength):
        """
        evaluates the opacity data.

        Inputs
        ------
        temperature: float
            The temperature to evaluate at.
        pressure: float
            The pressure to evaluate at.
        wavelength: float
            The wavelength to evaluate at.

        Outputs
        -------
        eval_result: float
            The evaluated opacity.
        """
        if temperature > self.T.max() or temperature < self.T.min():
            raise ValueError(
                f"Temperature {temperature} is outside the range of the data."
            )
        if pressure > self.P.max() or pressure < self.P.min():
            raise ValueError(f"Pressure {pressure} is outside the range of the data.")
        if wavelength > self.wl.max() or wavelength < self.wl.min():
            raise ValueError(
                f"Wavelength {wavelength} is outside the range of the data."
            )

        eval_result = self.eval_func(
            temperature,
            pressure,
            wavelength,
            self.T,
            self.P,
            self.wl,
            self.fitter_results,
            **self.eval_kwargs,
            **self.fit_kwargs,
        )
        # we need to recast the units if PLATON was used.
        if isinstance(self.load_obj, loader_platon):
            eval_result = 10**eval_result
            eval_result = eval_result / self.load_obj.species_weight / AMU / 1e-4
            if eval_result == 0:
                eval_result = 1e-104  # todo: fix this inf situation

        # todo: patch the eval neural net? it doesn't return the biases?

        return eval_result
