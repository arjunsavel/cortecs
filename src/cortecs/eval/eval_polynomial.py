import jax
import numpy as np

@jax.jit
def eval_polynomial(T, P, xsec, j):
    """
    evaluates the polynomial at a given temperature and pressure.
    :param T:
    :param P:
    :param xsec:
    :param j: the gas index to be used.
    :return:

    """
    TT = np.log10(T)
    PP = np.log10(P)

    z2 = (1.0, TT, PP, TT ** 2, TT ** 2 * PP, TT ** 2 * PP ** 2, TT * PP ** 2, TT * PP, TT * PP ** 3, PP ** 4, PP ** 5,
          PP ** 6, PP ** 7, PP ** 12, TT * PP ** 9, TT ** .25)
    Nterms = len(z2)

    xsec_int = 0.0
    for k in range(Nterms):
        xsec_int += z2[k] * xsec[k, j]
    xsec_int = 10 ** xsec_int  # we were keeping track of log opacity!

    return xsec_int