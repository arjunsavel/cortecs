import jax
import numpy as np


@jax.jit
def eval_pca_ind_wav(temperature_ind, pressure_ind, vectors, pca_coeffs):
    """
    Evaluates the PCA fit at a given temperature and pressure.

    Unfortunately, not all GPUs will support a simpler dot product, I believe,
    Also, we cannot loop over n_components explicitly because JAX
    functions require static loop lengths.

    Inputs
    ------
    temperature_ind: int
        The index of the temperature to evaluate at.
    pressure_ind: int
        The index of the pressure to evaluate at.
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
        xsec_val += (
            vectors[temperature_ind, component] * pca_coeffs[component, pressure_ind]
        )
    return xsec_val


def eval_pca(temperature, pressure, wavelength, T, P, wl, fitter_results):
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
    temperature_ind = np.where(np.isclose(T, temperature))[0][0]
    pressure_ind = np.where(np.isclose(P, pressure))[0][0]
    wavelength_ind = np.where(np.isclose(wl, wavelength))[0][0]
    pca_vectors, pca_coeffs_all_wl = fitter_results
    pca_coeffs = pca_coeffs_all_wl[wavelength_ind, :, :]

    return eval_pca_ind_wav(temperature_ind, pressure_ind, pca_vectors, pca_coeffs)
