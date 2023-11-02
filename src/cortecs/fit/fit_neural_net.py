"""
Trains a neural network to fit the opacity data.
"""
import pickle

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from cortecs.fit.metric_plot import *


# todo: loop over all wavelengths.
def prep_neural_net(cross_section):
    pass


def unravel_data(x, y, z=None, tileboth=False):
    """
    unravels the data into a single column. takes log as well.

    todo: move to a utils file?

    Inputs
    ------
    x: array-like
        first dimension of data.
    y: array-like
        second dimension of data.
    z: array-like
        third dimension of data.
    """

    if type(z) == type(None):
        if tileboth:
            return np.tile(np.log10(x), len(y))
        return np.repeat(np.log10(x), len(y))
    if tileboth:
        return np.tile(np.tile(np.log10(x), len(y)), len(z))

    return np.tile(np.repeat(np.log10(x), len(y)), len(z))


def fit_neural_net(
    cross_section,
    P,
    T,
    wl,
    all_wl=False,
    n_layers=3,
    n_neurons=8,
    activation="sigmoid",
    learn_rate=0.04,
    loss="mean_squared_error",
    epochs=2000,
    verbose=0,
    sequential_model=None,
    plot=False,
):
    """
    trains a neural network to fit the opacity data.
    :param Opac: not an actual opac.
    :param n_layers:
    :param n_neurons:
    :param activation:
    :param learn_rate:
    :param loss:
    :param epochs:
    :param verbose:
    :param sequential_model: if not None (and instead a sequential object), this overwrites the other neural net
                            parameters.
    :return:
        :history: the history object from the keras fit method
        :neural_network: the trained neural network
    """
    if sequential_model is None:
        layers_list = []
        for i in range(n_layers):
            layers_list += [layers.Dense(n_neurons, activation=activation)]

        layers_list += [layers.Dense(1)]  # final layer to predict single value

        neural_network = keras.Sequential(layers_list)

    else:
        neural_network = sequential_model

    # running the legacy Adam optimizer for the Mac folks!
    neural_network.compile(
        loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learn_rate)
    )

    # unpack opacity data. todo: not repeat this for every wavelength!
    if all_wl:
        raise NotImplementedError("todo: implement all_wl")
    P_unraveled = unravel_data(P, T, wl)
    T_unraveled = unravel_data(T, P, wl, tileboth=True)

    # todo: try to predict everything in one big net, not wavelength-independent?
    input_array = np.column_stack([P_unraveled, T_unraveled])
    predict_data_flattened = cross_section.flatten()

    history = neural_network.fit(
        input_array,
        predict_data_flattened,  # totally fine to overfit! we're not extrapolating, nor are we interpolating.
        verbose=verbose,
        epochs=epochs,
    )

    if plot:
        plot_loss(history)

    return history, neural_network


def save_neural_network(neural_network, filename):
    """
    saves the neural network to a file.
    :param neural_network:
    :param filename:
    :return:
    """
    n_layers = len(neural_network.layers)
    all_weights = []
    all_biases = []

    for layer in range(n_layers):
        layer_weights = neural_network.layers[layer].get_weights()[0]
        layer_biases = neural_network.layers[layer].get_weights()[1]
        all_weights += [layer_weights]
        all_biases += [layer_biases]

    # save all_weights and all_biases to a pickle
    with open(filename, "wb") as f:
        pickle.dump([all_weights, all_biases], f)

    # now, let's read it back in and make sure it's the same.

    return
