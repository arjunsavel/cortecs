import jax
import numpy as np

# todo: loop over all wavelengths.


def prep_polynomial(cross_section, wavelength_ind):
    pass


@jax.jit
def fit_polynomial(wavelength_ind, plot=False, species="H2", save=False):
    """
    todo: actually take in the Opac object.
    fits a polynomial to the opacity data.
    :param wavelength_ind:
    :param plot:
    :param species:
    :param save:
    :return:
    """
    xsecarr = species_dict[species]

    single_pres_single_temp = xsecarr[:, :, wavelength_ind].copy()
    X, Y = np.meshgrid(np.log10(T), np.log10(P), copy=False)
    #     X, Y = np.meshgrid(T, P, copy=False)
    Z = single_pres_single_temp

    X = X.flatten()
    Y = Y.flatten()

    # took out the y**2 term

    A = np.array(
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

    TT, PP = np.meshgrid(np.log10(T), np.log10(P), copy=False)

    z2 = (
        np.ones_like(TT),
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
        np.power(TT, 1 / 4),
    )

    im = np.tensordot(z2, coeff, axes=([0], [0]))

    return np.max(abs((im - Z) / Z)), np.mean(abs((im - Z) / Z)), coeff
