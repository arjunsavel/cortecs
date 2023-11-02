import jax
import jax.numpy as jnp
import numpy as np

# todo: loop over all wavelengths.


def prep_polynomial(cross_section, **kargs):
    pass


# @jax.jit
def fit_polynomial(Z, P, T, prep_res, plot=False, save=False):
    """
    todo: actually take in the Opac object.
    fits a polynomial to the opacity data.
    :param wavelength_ind:
    :param plot:
    :param species:
    :param save:
    :return:
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

    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=-1)

    return coeff
