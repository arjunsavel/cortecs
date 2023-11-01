import jax
import numpy as np


@jax.jit
def eval_polynomial(T, P, xsec, j):
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
    TT = np.log10(T)
    PP = np.log10(P)

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
        xsec_int += z2[k] * xsec[k, j]
    xsec_int = 10**xsec_int  # we were keeping track of log opacity!

    return xsec_int
