"""
Trains a neural network to fit the opacity data.
"""
import pickle
import warnings  # optional import handling should raise a warning instead of an error

import numpy as np

try:
    import keras
    import tensorflow as tf
    from tensorflow.keras import layers
except ModuleNotFoundError:
    warnings.warn(
        "The optional neural network-related packages have not been installed. If you would like to"
        + "use cortecs to fit with a neural network, please install cortecs with neural network functionality"
        + "as follows: python3 -m pip install -e .[neural_networks]"
    )

    # Set the optional modules to None, will raise an error if the user try to use tensorflow, but they have been warned
    keras = None
    tf = None
    layers = None
try:
    # running the legacy Adam optimizer for the Mac folks!
    from tensorflow.keras.optimizers.legacy import Adam

    Adam(0.1)  # test if it works
except ImportError:
    from tensorflow.keras.optimizers import Adam
from cortecs.fit.metrics import *


# todo: loop over all wavelengths.
def prep_neural_net(cross_section):
    pass


def unravel_data(x, y, z=None, tileboth=False):
    """
    unravels the data into a single column. takes log the log of quantities as well.

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

    if isinstance(z, type(None)):
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

    Inputs
    -------
        :cross_section: (ntemp x npressure) the array of cross-sections being fit.
        :P: (array) pressure grid corresponding to the cross-sections.
        :T: (array) temperature grid corresponding to the cross-sections.
        :wl: (array) wavelength grid corresponding to the cross-sections.
        :n_layers: (int) number of layers in the neural network. Increasing this number increases the flexibility
                of the model, but it also increases the number of parameters that need to be fit — and hence the model
                size and training time.
        :n_neurons: (int) number of neurons in each layer. Increasing this number increases the flexibility
                of the model, but it also increases the number of parameters that need to be fit — and hence the model
                size and training time.
        :activation: (str) activation function to use. See the keras documentation for more information:
                https://keras.io/api/layers/activations/
        :learn_rate: (float) learning rate for the optimizer. Increasing this number can help the model converge faster,
                but it can also cause the model to diverge.
        :loss: (str) loss function to use. See the keras documentation for more information:
                https://keras.io/api/losses/
        :epochs: (int) number of epochs to train for. Increasing this number can help the model converge better,
                but it primarily makes the training time longer.
        :verbose: (int) verbosity of the training.
        :sequential_model: (keras.Sequential) if not None, this is the neural network to be trained.
        :plot: (bool) whether to plot the loss.


    Returns
    -------
        :history: the history object from the keras fit method
        :neural_network: the trained neural network

    todo: implement all_wl
    """

    if sequential_model is None:
        layers_list = []
        for i in range(n_layers):
            layers_list += [layers.Dense(n_neurons, activation=activation)]

        layers_list += [layers.Dense(1)]  # final layer to predict single value

        neural_network = keras.Sequential(layers_list)

    else:
        neural_network = sequential_model

    neural_network.compile(loss=loss, optimizer=Adam(learn_rate))

    # unpack opacity data. todo: not repeat this for every wavelength!
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


def save_neural_net(filename, fit_results):
    """
    saves the neural network to a file.


    Inputs
    -------
        :filename: (str) filename to save the neural network to.
        :fit_results: (keras.Sequential) the neural network to save.


    Returns
    -------
        :None:
    """
    neural_network = fit_results[1]
    n_layers = len(neural_network.layers)
    all_weights = []
    all_biases = []

    for layer in range(n_layers):
        layer_weights = neural_network.layers[layer].get_weights()[0]
        layer_biases = neural_network.layers[layer].get_weights()[1]
        all_weights += [layer_weights]
        all_biases += [layer_biases]

    # save all_weights and all_biases to a pickle
    with open(filename + ".pkl", "wb") as f:
        pickle.dump([all_weights, all_biases], f)

    # now, let's read it back in and make sure it's the same.

    return
