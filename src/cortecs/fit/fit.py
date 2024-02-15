"""
The high-level API for fitting. Requires the Opac object.
"""
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm.notebook import tqdm

from cortecs.fit.fit_neural_net import *
from cortecs.fit.fit_pca import *
from cortecs.fit.fit_polynomial import *


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

    prep_method_dict = {
        "pca": prep_pca,
        "neural_net": prep_neural_net,
        "polynomial": prep_polynomial,
    }

    save_method_dict = {
        "pca": save_pca,
        "neural_net": save_neural_net,
        "polynomial": save_polynomial,
    }

    def __init__(self, opac, method="pca", **fitter_kwargs):
        """
        todo: make list of opac
        fits the opacity data to a neural network.

        Inputs
        ------
        opac: Opac
            the opacity object.
        method: str
            the method to use for fitting. Options include (in order of increasing complexity) include
            'polynomial', 'pca', and 'neural_net'. The more complex the model, the larger the model size (i.e., potentially
            the lower the compression factor), and the more likely it is to fit well.
        fitter_kwargs: dict
            kwargs that are passed to the fitter. one kwarg, for instance, is the fit_axis: for PCA,
            this determines what axis is fit against.
        """
        self.opac = opac
        self.fitter_kwargs = fitter_kwargs

        if method not in self.method_dict.keys():
            raise ValueError("method {} not supported".format(method))

        self.method = method

        self.fit_func = self.method_dict[self.method]

        self.wl = self.opac.wl
        self.P = self.opac.P
        self.T = self.opac.T

        # todo: figure out how to change the fitting...based on the fit axis?
        return

    def fit(self, parallel=False, verbose=1):
        """
        fits the opacity data to a neural network. loops over all wavelengths.
        Can loop over a list of species? ...should be parallelized!

        Inputs
        ------
        parallel: bool
            whether to parallelize the fitting.
        """

        # iterate over the wavelengths.
        prep_method = self.prep_method_dict[self.method]
        self.prep_res = prep_method(self.opac.cross_section, **self.fitter_kwargs)
        if not parallel:
            self.fit_serial(verbose=verbose)
        else:
            self.fit_parallel()

        return  # will I need to save and stuff like that?

    def fit_serial(self, verbose=0):
        """
        fits the opacity data with a given method. Serial.
        :return:
        todo: keep in mind that the PCA method is not actually independent of wavelength.
        """

        # loop over the wavelengths and fit
        res = []
        with warnings.catch_warnings():
            if verbose == 1:
                iterator = tqdm(enumerate(self.wl), total=len(self.wl), position=0, leave=True)
            else:
                iterator = enumerate(self.wl)
            for i, w in iterator:
                cross_section = self.opac.cross_section[:, :, i]
                res += [
                    self.fit_func(
                        cross_section,
                        self.P,
                        self.T,
                        self.prep_res,
                        **self.fitter_kwargs
                    )
                ]

        self.fitter_results = [self.prep_res, np.array(res)]

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
            num_processes = 1

            func = partial(
                self.fit_func,
                P=self.P,
                T=self.T,
                prep_res=self.prep_res,
                **self.fitter_kwargs
            )

            self.pbar = tqdm(
                total=len(self.wl),
                position=0,
                leave=True,
                unit="chunk",
                desc="Fitting with {} method".format(self.method),
            )

            # these two lines are where the bulk of the multiprocessing happens
            pool = Pool(num_processes)

            # actualy loop over using pool.map. need
            # reformat the cross_section to be a list of 2D arrays
            cross_section_reformatted = [
                self.opac.cross_section[:, :, i] for i in range(len(self.wl))
            ]

            # we tehcnically want sorted results. but apply async is the only way to get the progress bar to work!
            async_result = []
            for i, item in enumerate(cross_section_reformatted):
                async_result.append(
                    [i, pool.apply_async(func, args=(item,), callback=self.update_pbar)]
                )

            # Close the pool
            pool.close()
            pool.join()

            # Close the progress bar
            self.pbar.close()

            # sort the results based on the index
            sorted_results = [None] * len(async_result)
            for item in async_result:
                i, res = item
                sorted_results[i] = res.get()

        self.fitter_results = [self.prep_res, sorted_results]

        return

    def save(self, savename):
        """
        saves the fitter results.
        """
        save_func = self.save_method_dict[self.method]
        save_func(savename, self.fitter_results)
        return
