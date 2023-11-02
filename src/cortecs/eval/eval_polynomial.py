import jax
import jax.numpy as jnp


@jax.jit
def eval_polynomial(
    temperature,
    pressure,
    wavelength,
    temperatures,
    pressures,
    wavelengths,
    fitter_results,
):
    """
    evaluates the polynomial at a given temperature and pressure.

    Inputs
    ------
    T: float
        The temperature to evaluate at.
    P: float
        The pressure to evaluate at.
    xsec: array
        The cross section coefficients.
    j: int
        The index of the wavelength to evaluate at.

    Returns
    -------
    xsec_int: float
        The cross section at the given temperature and pressure.

    """
    TT = jnp.log10(temperature)
    PP = jnp.log10(pressure)
    coeffs = fitter_results[1]

    j = jnp.argmin(jnp.abs(wavelengths - wavelength))

    z2 = (
        1.0,
        TT,
        PP,
        TT**2,
        TT**2 * PP,
        TT**2 * PP**2,
        TT * PP**2,
        TT * PP,
        TT * PP**3,
        PP**4,
        PP**5,
        PP**6,
        PP**7,
        PP**12,
        TT * PP**9,
        TT**0.25,
    )
    Nterms = len(z2)

    xsec_int = 0.0
    for k in range(Nterms):
        xsec_int += z2[k] * coeffs[j, k]

    return xsec_int
