"""
Performs simple optimization of PCA hyperparameters — i.e., number of components and wavelength index
for computing eigenvectors.
"""
import math

import numpy as np
from tqdm.autonotebook import tqdm

from cortecs.fit.fit import Fitter
from cortecs.fit.fit_neural_net import *
from cortecs.fit.fit_pca import *


def optimize_pca(
    max_size,
    max_evaluations,
    opac,
    min_components=3,
    max_components=5,
    wav_ind_start=3573,
):
    """

    Inputs
    ------
    max_size: float
        maximum size of file in kB.
    max_evaluations: int
        maximum number of evaluations of the fitter



    """
    T = opac.T
    P = opac.P
    wl = opac.wl
    cross_section = opac.cross_section
    # each axis — the wavelength index being tested and the number of components — will be tested n times.
    # n * n = max_evaluations.
    n_test_each_axis = math.floor(np.power(max_evaluations, 1 / 2))

    n_pc_range = np.linspace(min_components, max_components, n_test_each_axis).astype(
        int
    )
    wav_ind_range = np.linspace(wav_ind_start, len(wl) - 1, n_test_each_axis).astype(
        int
    )
    print("len wl")
    print(len(wl))
    print("wl range")
    print(wav_ind_range)
    (
        n_pc_grid,
        wav_ind_grid,
    ) = np.meshgrid(n_pc_range, wav_ind_range)
    # max_size currently isn't used.
    final_errors = []
    lin_samples = []
    # ah. we're supposed to fit at every wavelength.
    for sample in tqdm(
        range(len(n_pc_grid.flatten())),
        desc="Optimizing PCA hyperparameters",
    ):
        n_pc, wav_ind = (
            n_pc_grid.flatten()[sample],
            wav_ind_grid.flatten()[sample],
        )
        fitter = Fitter(opac, method="pca", wav_ind=wav_ind, nc=n_pc)
        try:
            fitter.fit(verbose=0)

            # evaluate the fit
            vals, orig_vals, abs_diffs, percent_diffs = calc_metrics(fitter, plot=False)
            mse = np.mean(np.square(abs_diffs))

        except ValueError as e:
            mse = np.inf

        final_errors += [mse]
        lin_samples += [sample]

    # return the best-performing hyperparameters
    best_sample_ind = lin_samples[np.argmin(final_errors)]
    best_params = {
        "n_pc": n_pc_grid.flatten()[best_sample_ind],
        "wav_ind": wav_ind_grid.flatten()[best_sample_ind],
    }
    return best_params
