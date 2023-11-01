"""
Plots histograms and stuff after the fitting has occurred so that the level of fit can be assessed.
"""
import matplotlib.pyplot as plt
import numpy as np


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


def plot_quantity(species_quantity_dict, quantity="mean", log=True, nbins=20):
    """
    plots the quantities related to the species fitting.

    Inputs
    ------
    species_quantity_dict: dict
        dictionary of species and the quantities associated with them.
    quantity: str
        quantity to plot. 'mean', 'std', 'min', 'max', 'median', 'percentile'
    log: bool
        whether to plot the log of the quantity.
    nbins: int
        number of bins to use in the histogram.

    """
    # todo: colormaps

    # find the bins
    min = np.inf
    max = -np.inf
    for species in species_quantity_dict.keys():
        species_min = np.min(species_quantity_dict[species])
        min = np.min([min, species_min])

        species_max = np.max(species_quantity_dict[species])
        max = np.max([min, species_max])

    if log:
        bins = np.logspace(np.log10(min), np.log10(max), nbins)
    else:
        bins = np.linspace(min, max, nbins)

    for species in species_quantity_dict.keys():
        if log:
            quantity = np.log10(species_quantity_dict[species])
        else:
            quantity = species_quantity_dict[species]

        plt.hist(quantity, label=species, lw=5, histtype="step", bins=bins)

    plt.legend()
    plt.title("Fitting quality: " + quantity)
