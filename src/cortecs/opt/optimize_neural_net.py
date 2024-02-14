"""
Performs simple optimization of neural network hyperparameters.
"""
import math

import numpy as np
from tqdm.autonotebook import tqdm

from cortecs.fit.fit_neural_net import *


def optimize_neural_net(
    max_size,
    max_evaluations,
    opac,
    min_layers=2,
    min_neurons=2,
    max_layers=3,
    max_neurons=13,
    min_learn_rate=0.01,
    max_learn_rate=0.1,
):
    """
    performs simple optimization of neural network hyperparameters.

    Inputs
    ------
    max_size: float
        maximum size of file in kB.
    max_evaluations: int
        maximum number of evaluations of the neural network.
    """
    cross_section = opac.cross_section
    T = opac.T
    P = opac.P
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be greater than 0.")

    # what are the things we'll be changing? the size of the neural network,
    # the number of layers, the number of neurons per layer, the activation function, the learning rate...

    # restrict based on max size

    # 1.1 kB for 3 layers, 8 nodes / layer =216 weights
    n_weights = 216 * max_size / 1.1

    # let's say it's a fully connected neural network with a bias term. then, for a single layers with m neurons each, there are
    # m * (m + 1) = m^2 + m weights. so, for n layers, there are n * (m^2 + m) weights. So we basically want to find
    # the biggest n, m to satisfy n * (m^2 + m) < n_weights.
    # this grows faster with m.

    n_steps = math.floor(np.power(max_evaluations, 1 / 4))
    n_weights = max_neurons * (max_neurons + 1) * max_layers

    print("max number of weights: ", n_weights)

    n_layers_range = np.linspace(min_layers, max_layers, n_steps).astype(int)
    n_neurons_range = np.linspace(min_neurons, max_neurons, n_steps)
    activation_range = ["sigmoid", "relu"]
    learn_rate_range = np.geomspace(min_learn_rate, max_learn_rate, n_steps)

    print("max number of layers: ", max_layers)
    print("max number of neurons / layer: ", max_neurons)

    n_layers_grid, n_neurons_grid, activation_grid, learn_rate_grid = np.meshgrid(
        n_layers_range, n_neurons_range, activation_range, learn_rate_range
    )
    final_losses = []
    lin_samples = []
    for sample in tqdm(
        range(len(n_layers_grid.flatten())),
        desc="Optimizing neural network hyperparameters",
    ):
        n_layers, n_neurons, activation, learn_rate = (
            n_layers_grid.flatten()[sample],
            n_neurons_grid.flatten()[sample],
            activation_grid.flatten()[sample],
            learn_rate_grid.flatten()[sample],
        )
        history, _ = fit_neural_net(
            cross_section[:, :, -2],
            T,
            P,
            None,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            learn_rate=learn_rate,
            loss="mean_squared_error",
            epochs=2000,
            verbose=0,
            sequential_model=None,
            plot=False,
        )
        final_loss = history.history["loss"][-1]
        final_losses += [final_loss]
        lin_samples += [sample]

    # return the best-performing hyperparameters
    best_sample_ind = lin_samples[np.argmin(final_losses)]
    best_params = {
        "n_layers": n_layers_grid.flatten()[best_sample_ind],
        "n_neurons": n_neurons_grid.flatten()[best_sample_ind],
        "activation": activation_grid.flatten()[best_sample_ind],
        "learn_rate": learn_rate_grid.flatten()[best_sample_ind],
    }
    return best_params
