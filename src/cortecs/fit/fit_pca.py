"""
Module for performing PCA regression on opacity functions.
"""

import numpy as np


def standardize_cube(input_array):
    """
    Prepares an array for PCA.

    Inputs
    -------
        :input_array: (ntemperature x npressure) array for PCA.

    Returns
    -------
        :standardized_cube: (ntemperature x npressure) standardized (mean-substracted, standard deviation-divided) array for PCA.
    """
    nf, nx = input_array.shape
    mat = input_array.copy()

    if np.any(np.isnan(mat)):
        raise ValueError("NaNs in input array.")
    # loop over the first index
    for i in range(nf):
        mat[i, :] -= np.mean(mat[i, :])
        mat[i, :] /= np.std(mat[i, :])
    standardized_cube = mat
    return standardized_cube


def do_svd(standardized_cube, nc, nx):
    """
    Does the SVD (singular value decomposition) on the standardized cube.

    Inputs
    -------
        :standardized_cube: (ntemperature x npressure) standardized (mean-substracted, standard deviation-divided) array for PCA.
        :nc: (int) number of PCA components to keep in the reconstruction. Increasing the number of components can make the reconstruction
        of opacity data more accurate, but it can also lead to overfitting and make the model size larger (i.e.,
        decrease the compression factor).

    """

    xMat = np.ones((nx, nc + 1))  # Note the extra column of 1s
    mat = np.moveaxis(standardized_cube.copy(), 0, -1)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    xMat[:, 1:] = u[:, 0:nc]  # The first column is left as a column of ones

    return xMat, s, vh, u


def fit_mlr(cube, X):
    """
    todo: check array shapes.
    Fits the MLR (multiple linear regression) to the input cube with the PCA outputs and normalizes accordingly.

    Inputs
    -------
        :cube: (ntemp x npressure) array being fit with PCA. Generally, this is the opacity data at a single
        wavelength as a function of temperature and pressure.
        :X: (npressure x ncomponents) PCA components. these are set by functions such as `do_pca`.

    Returns
    -------
        :beta: (ncomponents x ntemp) PCA coefficients. These are the coefficients that are used to reconstruct the
        opacity data at a later stage (e.g., within an atmospheric forward model). They represent the "compressed" opacity.
    """
    Y = np.moveaxis(cube.copy(), 0, -1)
    term1 = np.linalg.inv(np.dot(X.T, X))
    term2 = np.dot(term1, X.T)
    beta = np.dot(term2, Y)
    return beta


def do_pca(cube, nc=3):
    """
    Does all the PCA steps. right now for a single spectrum.

    Inputs
    -------
        :cube: (ntemperature x npressure) array being fit with PCA. Generally, this is the opacity data at a single
        wavelength as a function of temperature and pressure.
        :nc: (int) number of PCA components to keep. Increasing the number of components can make the reconstruction
        of opacity data more accurate, but it can also lead to overfitting and make the model size larger (i.e.,
        decrease the compression factor).

    """

    # do the PCA
    standardized_cube = standardize_cube(cube)

    nf, nx = cube.shape
    """ Choosing the number of components and initialising the matrix of eigenvectors """

    try:
        xMat, s, vh, u = do_svd(standardized_cube, nc, nx)

    except np.linalg.LinAlgError as e:
        print("SVD did not converge.")
        raise e

    return xMat, standardized_cube, s, vh, u


def fit_pca(cross_section, P, T, prep_res, fit_axis="pressure", **kwargs):
    """
    Fits the PCA to the opacity data.

    Inputs
    -------
        :cross_section: (ntemp x npressure) the array of cross-sections being fit.
        :P: pressure grid
        :T: temperature grid
        :xMat: (npres x nc) PCA components

    Returns
    -------
        :beta: (nc x pixels) PCA coefficients
    """
    # print("shapes for everything:", cross_section.shape, P.shape, T.shape, xMat.shape)
    xMat = prep_res
    cross_section = move_cross_section_axis(cross_section, fit_axis)
    beta = fit_mlr(cross_section, xMat)
    return beta


def move_cross_section_axis(cross_section, fit_axis, dim=2):
    """
    todo: add docstring
    :param cross_section:
    :param fit_axis:
    :return:
    """
    fit_axis_options = ["best", "temperature", "pressure"]
    if fit_axis not in fit_axis_options:
        raise ValueError(f"fit_axis param must be one of: {fit_axis_options}")

    # the current shape prefers pressure. let's auto-check, though
    if fit_axis == "best":
        # actually want SECOND longest.
        if dim == 3:
            longest_axis = np.argmax(cross_section[:, :, 0].shape)
        else:
            longest_axis = np.argmax(cross_section.shape)

        # now move that longest axis to 0.
        cross_section = np.moveaxis(cross_section, longest_axis, 1)

    elif fit_axis == "temperature":
        cross_section = np.moveaxis(cross_section, 0, 1)

    return cross_section


def prep_pca(
    cross_section, wav_ind=-1, nc=2, force_fit_constant=False, fit_axis="pressure"
):
    """
    Prepares the opacity data for PCA. That is, it calculates the PCA components to be fit along the entire
    dataset by fitting the PCA to a single wavelength.

    todo: perform type checking for inputs.

    Inputs
    -------
        :cross_section: (ntemp x npressure x nwavelength) the array of cross-sections being fit.
        :wav_ind: (int) index of wavelength to fit. Ideally, this should be fit to a wavelength that has somewhat
        representative temperature--pressure structure to the opacity function. If there is *no* temperature or wavelength
        dependence at this wavelength index but strong dependence elsewhere, for instance, the opacity function will
        be poorly reconstructed.
        :nc: (int) number of PCA components to keep. Increasing the number of components can make the reconstruction
        of opacity data more accurate, but it can also lead to overfitting and make the model size larger (i.e.,
        decrease the compression factor).
        :force_fit_constant: (bool) if True, will allow the PCA to fit an opacity function without temperature and pressure
        dependence. This usually isn't recommended, if these PCA vectors are to e used to fit other wavelengths that
        *do* have temperature and pressure dependence.
        :fit_axis: (str) the axis to fit against. determines the shape of the final vectors and components.
        if "best", chooses the largest axis. otherwise, can select "temperature" or "pressure".

    Returns
    -------
        :xMat: (n_exp x nc) PCA components
    """
    cross_section = move_cross_section_axis(cross_section, fit_axis, dim=3)
    single_pres_single_temp = cross_section[:, :, wav_ind]
    if (
        np.all(single_pres_single_temp == single_pres_single_temp[0, 0])
        and not force_fit_constant
    ):
        raise ValueError(
            "all values are the same at this wavelength index! Try a different one."
        )

    xMat, fStd, s, vh, u = do_pca(single_pres_single_temp, nc)
    return xMat


def save_pca(savename, fit_results):
    """
    Saves the PCA components and coefficients to files

    Inputs
    -------
        :savename: (str) if not None, the PCA components will be saved to this filename.
        :fit_results: contains the PCA coeefficients and vectors.
    """
    # pdb.set_trace()
    vectors, beta = fit_results
    np.savez_compressed(savename, pca_coeffs=beta, vectors=vectors)
