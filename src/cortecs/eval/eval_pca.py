"""
For evaluating the PCA fit at a given temperature, pressure, and wavelength.
"""

import jax
import numpy as np


@jax.jit
def eval_pca_ind_wav(first_ind, second_ind, vectors, pca_coeffs):
    """
    Evaluates the PCA fit at a given temperature and pressure.

    Unfortunately, not all GPUs will support a simpler dot product, I believe,
    Also, we cannot loop over n_components explicitly because JAX
    functions require static loop lengths.

    Inputs
    ------
    first_ind: int
        The index of the first axis quantity (default temperature) to evaluate at.
    second_ind: int
        The index of the second axis quantity (default pressure) to evaluate at.
    vectors: array
        The PCA vectors.
    pca_coeffs: array
        The PCA coefficients.
    n_components: int
        The number of PCA components used in the fitting


    """

    xsec_val = 0.0
    n_components = vectors.shape[1]
    for component in range(n_components):
        xsec_val += vectors[first_ind, component] * pca_coeffs[component, second_ind]
    return xsec_val


def eval_pca(
    temperature,
    pressure,
    wavelength,
    T,
    P,
    wl,
    fitter_results,
    fit_axis="pressure",
    **kwargs
):
    """
    Evaluates the PCA fit at a given temperature, pressure, and wavelength.


    Inputs
    ------
    temperature: float
        The temperature to evaluate at.
    pressure: float
        The pressure to evaluate at.
    wavelength: float
        The wavelength to evaluate at.

    """
    # find the nearest temperature, pressure, and wavelength indices.

    temperature_ind = np.argmin(np.abs(T - temperature))
    pressure_ind = np.argmin(np.abs(P - pressure))
    wavelength_ind = np.argmin(np.abs(wl - wavelength))

    pca_vectors, pca_coeffs_all_wl = fitter_results
    pca_coeffs = pca_coeffs_all_wl[wavelength_ind, :, :]

    # todo: figure out how to order the pressure and temperature inds!
    # pdb.set_trace()
    if fit_axis == "pressure":
        first_arg = pressure_ind
        second_arg = temperature_ind
    elif fit_axis == "temperature":
        first_arg = temperature_ind
        second_arg = pressure_ind

    elif fit_axis == "best":
        T_length = len(T)
        P_length = len(P)

        # todo: what if equal?
        if T_length > P_length:
            first_arg = temperature_ind
            second_arg = pressure_ind
        else:
            first_arg = pressure_ind
            second_arg = temperature_ind
    # print("first_arg, second_arg", first_arg, second_arg)
    # print("shapes:", pca_vectors.shape, pca_coeffs.shape)
    return eval_pca_ind_wav(first_arg, second_arg, pca_vectors, pca_coeffs)
