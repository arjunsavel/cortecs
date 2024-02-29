"""
Plots histograms and stuff after the fitting has occurred so that the level of fit can be assessed.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

from cortecs.eval.eval import Evaluator


def plot_loss(history):
    """
    plots the loss from the history object. need to substantially rewrite
    TODO: rcparams file to make things look nice, fontwise and such?
    :param history:
    :return:
    """
    plt.plot(history.history["loss"], label="loss", color="teal", lw=3)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Error", fontsize=20)  # need the units on this
    plt.yscale("log")
    plt.legend()
    return


def title_format(quantity):
    """
    formats the title of the histogram based on the quantity being plotted.

    Inputs
    ------
    quantity: array-like
        the quantity being plotted.
    """
    plt.title(
        "Mean: {:.2f}, Std: {:.2f}".format(np.mean(quantity), np.std(quantity)),
        fontsize=20,
    )
    return


def calc_metrics(fitter, tp_undersample_factor=1, wl_under_sample_factor=8, plot=False):
    """
        calculates the mean and percent error  associated with the fit.
    .
        Inputs
        ------
        opac_obj: Opac
            the opacity object to be assessed.
        fitter: Fitter
            the fitter object that was used to fit the opacity object.
        tp_undersample_factor: int
            factor to undersample the temperature and pressure grids by when calculating the fit metrics
        wl_under_sample_factor: int
            factor to undersample the wavelength grid by.
    """
    evaluator = Evaluator(fitter.opac, fitter)
    AMU = 1.6605390666e-24  # atomic mass unit in cgs. From astropy!

    if not hasattr(evaluator.load_obj, "species_weight"):
        species_weight = 1
    else:
        species_weight = evaluator.load_obj.species_weight

    vals = []
    orig_vals = []
    abs_diffs = []
    percent_diffs = []
    for i in tqdm(range(len(fitter.opac.T))[::tp_undersample_factor]):
        for j in range(len(fitter.opac.P))[::tp_undersample_factor]:
            for k in range(len(fitter.opac.wl))[::wl_under_sample_factor]:
                val = evaluator.eval(
                    fitter.opac.T[i], fitter.opac.P[j], fitter.opac.wl[k]
                )
                # todo: check that this works for not just PLATON
                val = np.log10(val * species_weight * AMU * 1e-4)
                if not np.isfinite(val):
                    val = -104
                orig_val = fitter.opac.cross_section[i, j, k]
                abs_diffs += [val - orig_val]
                percent_diffs += [100 * (val - orig_val) / orig_val]
                vals += [val]
                orig_vals += [orig_val]
    abs_diffs = np.array(abs_diffs)
    percent_diffs = np.array(percent_diffs)
    orig_vals = np.array(orig_vals)
    vals = np.array(vals)

    if plot:
        plt.figure()
        plt.hist(abs_diffs, color="dodgerblue")
        plt.xlabel("Abs. residuals in log10 opacity", fontsize=20)
        plt.ylabel("Count", fontsize=20)

        title_format(abs_diffs)

        plt.figure()
        plt.hist(percent_diffs, color="goldenrod")
        plt.xlabel("Percent. residuals in log10 opacity", fontsize=20)
        title_format(abs_diffs)

        plt.ylabel("Count", fontsize=20)

    return vals, orig_vals, abs_diffs, percent_diffs
