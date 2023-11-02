"""
for optimizing the fits.
"""

from cortecs.opt.optimize_neural_net import *
from cortecs.opt.optimize_pca import *
from cortecs.opt.optimize_polynomial import *


class Optimizer(object):
    """
    for optimizing the fits.
    """

    method_dict = {
        "pca": optimize_pca,
        "neural_net": optimize_neural_net,
        "polynomial": optimize_polynomial,
    }

    def __init__(self, fitter, **optim_kwargs):
        """
        :param fitter: Fitter object
        :param optim_kwargs: kwargs for the optimizer
        """
        method = fitter.method
        self.opac = fitter.opac
        self.load_obj = fitter.load_obj
        self.wl = fitter.wl
        self.P = fitter.P
        self.T = fitter.T
        self.fitter = fitter
        self.optim_kwargs = optim_kwargs
        self.optim_func_kwargs = self.fitter.optim_func_kwargs

        self.opt_func = self.method_dict[method]
        return

    def optimize(self):
        """
        optimizes the fit.
        :return:
        """
        self.best_params = self.opt_func()
        return
