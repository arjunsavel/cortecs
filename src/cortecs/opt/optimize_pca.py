"""
Performs simple optimization of PCA hyperparameters — i.e., number of components and wavelength index
for computing eigenvectors.
"""
import math

import numpy as np
from tqdm import tqdm

from cortecs.fit.fit_neural_net import *


def optimize_pca(
    max_size,
    max_evaluations,
    cross_section,
    P,
    T,
    wl,
    min_components=2,
    max_components=4,
):
    """

    Inputs
    ------
    max_size: float
        maximum size of file in kB.
    max_evaluations: int
        maximum number of evaluations of the fitter



    """

    # each axis — the wavelength index being tested and the number of components — will be tested n times.
    # n * n = max_evaluations.
    n_test_each_axis = math.floor(np.power(max_evaluations, 1 / 2))

    n_pc_range = np.linspace(min_components, max_components, n_test_each_axis).astype(
        int
    )
    wav_ind_range = np.linspace(0, len(wl), n_test_each_axis).astype(int)

    (
        n_pc_grid,
        wav_ind_grid,
    ) = np.meshgrid(n_pc_range, wav_ind_range)

    # max_size currently isn't used.
    final_errors = []
    lin_samples = []
    for sample in tqdm(
        range(len(n_pc_range.flatten())),
        desc="Optimizing neural network hyperparameters",
    ):
        n_pc, wav_ind = (
            n_pc_grid.flatten()[sample],
            wav_ind_grid.flatten()[sample],
        )
        xMat = prep_pca(cross_section, wav_ind=wav_ind, nc=n_pc)
        fit_pca(cross_section, P, T, xMat, nc=3, wav_ind=1, savename=None)

        # evaluate the fit
        vals, orig_vals, abs_diffs, percent_diffs = calc_metrics(
            opac_obj, fitter, plot=True
        )
        mse = np.mean(np.square(abs_diffs))

        final_errors += [mse]
        lin_samples += [sample]

    # return the best-performing hyperparameters
    best_sample_ind = lin_samples[np.argmin(final_errors)]
    best_params = {
        "n_pc": n_pc_grid.flatten()[best_sample_ind],
        "wav_ind": wav_ind_grid.flatten()[best_sample_ind],
    }

    return best_params
