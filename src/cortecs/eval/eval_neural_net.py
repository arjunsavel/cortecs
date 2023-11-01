"""
This file holds the classes for evaluating opacity data as trained by a neural network.

maybe call all of these eval_poly.py, eval_neural_net.py, etc.?

author: @arjunsavel
"""
import jax
import numpy as np
from jax import lax


@jax.jit
def feed_forward(x, n_layers, weights, biases):
    """
    feed forward neural network. this is a function that takes in the input, weights, and biases and returns the output.

    This can probably be optimized, to be honest.
    :param x:
    :param weights:
    :param biases:
    :return:
    """

    def inner_function(i, x):
        return jax.nn.sigmoid(x.dot(weights[i]) + biases[i])

    res = lax.fori_loop(0, n_layers, inner_function, x)

    return res


@jax.jit
def eval_neural_net(x, n_layers, weights, biases):
    res = feed_forward(x[0], n_layers, weights, biases)
    return res
