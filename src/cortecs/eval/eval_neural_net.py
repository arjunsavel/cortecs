"""
This file holds the classes for evaluating opacity data as trained by a neural network.

maybe call all of these eval_poly.py, eval_neural_net.py, etc.?

author: @arjunsavel
"""
import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def feed_forward_equal_layer_sizes(x, n_layers, weights, biases):
    """
    feed forward neural network. this is a function that takes in the input, weights, and biases and returns the output.

    This function only works if all layers have the same size.

    Inputs
    ------
    x: array-like
        the input to the neural network.
    n_layers: int
        the number of layers in the neural network.
    weights: list
        the weights of the neural network.
    biases: list
    """

    def inner_function(i, x):
        return jax.nn.sigmoid(x.dot(weights[i]) + biases[i])

    res = lax.fori_loop(0, n_layers, inner_function, x)

    return res


# todo: add test to make sure feed forward actually works!
def feed_forward(x, n_layers, weights, biases):
    """
    feed forward neural network. this is a function that takes in the input, weights, and biases and returns the output.


    Inputs
    ------
    x: array-like
        the input to the neural network.
    n_layers: int
        the number of layers in the neural network.
    weights: list
        the weights of the neural network.
    biases: list
    """
    res = x
    for i in range(n_layers - 1):
        res = jax.nn.sigmoid(res.dot(weights[i]) + biases[i])
    res = res.dot(weights[-1]) + biases[-1]
    return res


# @jax.jit
def eval_neural_net(
    T,
    P,
    temperatures=None,
    pressures=None,
    wavelengths=None,
    n_layers=None,
    weights=None,
    biases=None,
    **kwargs
):
    """
    evaluates the neural network at a given temperature and pressure.

    Inputs
    ------
    T: float
        The temperature to evaluate at.
    P: float
        The pressure to evaluate at.
    n_layers: int
        The number of layers in the neural network.
    weights: list
        The weights of the neural network.
    biases: list
        The biases of the neural network.
    """
    x = jnp.array([jnp.log10(T), jnp.log10(P)])
    res = feed_forward(x, n_layers, weights, biases)
    return res
