"""
Trains a neural network to fit the opacity data.
"""
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# todo: loop over all wavelengths.


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
    verbose=1,
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

    neural_network.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learn_rate))

    # unpack opacity data. todo: not repeat this for every wavelength!
    P_unraveled = np.tile(np.repeat(np.log10(P), len(T)), len(wl))
    T_unraveled = np.tile(np.tile(np.log10(T), len(P)), len(wl))
    # lambda_unraveled = np.repeat(wl, len(P) * len(T))

    # todo: try to predict everything in one big net, not wavelength-independent?
    input_array = np.column_stack([T_unraveled, P_unraveled])
    predict_data_flattened = cross_section.flatten()

    history = neural_network.fit(
        input_array,
        predict_data_flattened,  # totally fine to overfit!
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

    for layer in n_layers:
        layer_weights = neural_network.layers[layer].get_weights()[0]
        layer_biases = neural_network.layers[layer].get_weights()[1]
        all_weights += [layer_weights]
        all_biases += [layer_biases]

    np.savez(filename, all_weights, all_biases)
    return


# todo: plot all!
