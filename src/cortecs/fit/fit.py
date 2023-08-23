"""
The high-level API for fitting. Requires the Opac function.
"""
import warnings
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from cortecs.fit_neural_net import *
from cortecs.fit_pca import *
from cortecs.fit_polynomial import *


class Fitter(object):
    """
    fits the opacity data to a neural network.

    todo: fit CIA. only on one dimension, because there's no pressure dependence.
    """

    # make the dictionary of stuff here
    method_dict = {
        "pca": fit_pca,
        "neural_net": fit_neural_net,
        "polynomial": fit_polynomial,
    }

    def __init__(self, Opac, method="pca", **fitter_kwargs):
        """
        todo: make list of opac
        fits the opacity data to a neural network.
        :param Opac: Opac object
        """
        self.Opac = Opac
        self.fitter_kwargs = fitter_kwargs
        self.method = method

        self.fit_func = self.method_dict[self.method]

        return

    def fit(self, parallel=False):
        """
        fits the opacity data to a neural network. loops over all wavelengths.
        Can loop over a list of species? ...should be parallelized!
        :param plot: whether or not to plot the loss
        :return: history, neural_network
        """

        # iterate over the wavelengths.
        if not parallel:
            self.fit_serial()
        else:
            self.fit_parallel()

        return  # will I need to save and stuff like that?

    def fit_serial(self):
        """
        fits the opacity data with a given method. Serial.
        :return:
        todo: keep in mind that the PCA method is not actually independent of wavelength.
        """
        # so maybe have some pre-fit function defined here for everything, but it only does something for PCA.

        # loop over the wavelengths and fit
        with warnings.catch_warnings():
            for i, w in tqdm(enumerate(self.wl), total=len(self.wl)):
                cross_section = self.Opac.cross_section[:, :, i]
                self.fit_func(cross_section, self.P, self.T, **self.fitter_kwargs)

        return

    def update_pbar(self, arg):
        """
        Updates a tqdm progress bar.
        """
        self.pbar.update(1)
        pass

    def fit_parallel(self):
        """
        fits the opacity data with a given method. Parallel.
        :return:
        """
        with warnings.catch_warnings():
            num_processes = 3
            func = partial(self.func_fit, self.P, self.T, **self.fitter_kwargs)

            self.pbar = tqdm(
                total=len(self.wl),
                position=0,
                leave=True,
                unit="chunk",
                desc="Fitting with {} method".format(self.method),
            )

            # these two lines are where the bulk of the multiprocessing happens
            p = Pool(num_processes)

            for i in range(len(self.wl)):
                p.apply_async(
                    func,
                    args=(self.Opac.cross_section[:, :, i],),
                    callback=self.update_pbar,
                )
            p.close()
            p.join()
            self.pbar.close()

        return
