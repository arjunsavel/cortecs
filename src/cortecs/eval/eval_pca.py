import jax


@jax.jit
def eval_pca(temperature_ind, pressure_ind, vectors, pca_coeffs):
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
