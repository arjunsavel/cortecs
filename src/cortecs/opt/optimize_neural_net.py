"""
Performs simple optimization of neural network hyperparameters.
"""
import numpy as np

from cortecs.fit.fit_neural_net import *


def optimize_neural_net(
    max_size,
    max_evaluations,
    cross_section,
    P,
    T,
    wl,
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
        maximum size of file in MB.
    max_evaluations: int
        maximum number of evaluations of the neural network.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be greater than 0.")

    # what are the things we'll be changing? the size of the neural network,
    # the number of layers, the number of neurons per layer, the activation function, the learning rate...

    # restrict based on max size
    n_weights = (
        max_size / 1e3
    )  # there are 1e3 nodes in a MB. (that's not true, but it's just to play with for now)

    # let's say it's a fully connected neural network with a bias term. then, for a single layers with m neurons each, there are
    # m * (m + 1) = m^2 + m weights. so, for n layers, there are n * (m^2 + m) weights. So we basically want to find
    # the biggest n, m to satisfy n * (m^2 + m) < n_weights.
    # this grows faster with m.

    n_steps = round(np.power(max_evaluations, 1 / 4)) // 2
    n_weights = max_neurons * (max_neurons + 1) * max_layers

    print("max number of weights: ", n_weights)

    n_layers_range = np.linspace(min_layers, max_layers, n_steps)
    n_neurons_range = np.linspace(min_neurons, max_neurons, n_steps)
    activation_range = ["sigmoid", "relu"]
    learn_rate_range = np.geomspace(min_learn_rate, max_learn_rate, n_steps)

    n_layers_grid, n_neurons_grid, activation_grid, learn_rate_grid = np.meshgrid(
        n_layers_range, n_neurons_range, activation_range, learn_rate_range
    )
    final_losses = []
    lin_samples = []
    for sample in zip(n_layers_grid, n_neurons_grid, activation_grid, learn_rate_grid):
        n_layers, n_neurons, activation, learn_rate = sample
        history, _ = fit_neural_net(
            cross_section,
            P,
            T,
            wl,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation="sigmoid",
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

    # return the best-performing hyperparameters.
    return lin_samples[np.argmin(final_losses)]
