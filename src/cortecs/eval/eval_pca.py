import jax


@jax.jit
def eval_pca(temperature_ind, pressure_ind, vectors, pca_coeffs, n_components=3):
    """
    Evaluates the PCA fit at a given temperature and pressure.

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
    for component in range(n_components):
        xsec_val += (
            vectors[temperature_ind, component] * pca_coeffs[component, pressure_ind]
        )
    return xsec_val
