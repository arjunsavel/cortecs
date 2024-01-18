"""
Module for fitting polynomial functions in temperature and pressure to opacity functions.
"""

import jax.numpy as jnp
import numpy as np

# todo: loop over all wavelengths.


def prep_polynomial(cross_section, **kargs):
    pass


# @jax.jit
def fit_polynomial(
    Z,
    P,
    T,
    prep_res,
    plot=False,
):
    """
    fits a polynomial to the opacity data.

    Inputs
    -------
        :Z: (n_temp x n_pres) it's an array.
        :P: pressure grid
        :T: temperature grid
        :prep_res: (n_temp x n_pres) PCA components
        :plot: (bool) whether to plot the fit.
        :savename: (str) if not None, the PCA components will be saved to this filename.

    Returns
    -------
        :coeff: (nc x pixels) PCA coefficients
    """
    X, Y = jnp.meshgrid(jnp.log10(T), jnp.log10(P), copy=True)

    X = X.flatten()
    Y = Y.flatten()

    # took out the y**2 term

    A = jnp.array(
        [
            X * 0 + 1,
            X,
            Y,
            X**2,
            X**2 * Y,
            X**2 * Y**2,
            X * Y**2,
            X * Y,
            X * Y**3,
            Y**4,
            Y**5,
            Y**6,
            Y**7,
            Y**12,
            X * Y**9,
            np.power(X, 1 / 4),
        ]
    ).T

    B = Z.flatten()

    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)

    return coeff


def save_polynomial(savename, coeff):
    """
    Saves the polynomial coefficients to a file.

    Inputs
    -------
        :savename: (str) if not None, the PCA components will be saved to this filename.
        :coeff: (nc x ntemp) PCA coefficients
    """
    np.savetxt(savename + ".txt", coeff)
