import warnings

import numpy as np


def standardize_cube(input_array):
    """
    Prepares an array for PCA.

    Inputs
    -------
        :input_array: (n_exp x pixels) it's an array.

    Returns
    -------
        :standardized_cube: (n_exp x pixels) standardized array for PCA.
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
    Does the SVD on the standardized flux cube.

    """

    xMat = np.ones((nx, nc + 1))  # Note the extra column of 1s
    mat = np.moveaxis(standardized_cube.copy(), 0, -1)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    xMat[:, 1:] = u[:, 0:nc]  # The first column is left as a column of ones

    return xMat, s, vh, u


def fit_mlr(cube, X):
    """
    Fits the MLR to the flux cube with the PCA outputs and normalizes accordingly.
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
        :wav_for_pca: (n_exp x pixels) it's an array.
        :flux_for_pca: it's the same.
        :nc: (int) number of PCA components to keep.

    """

    # do the PCA
    standardized_cube = standardize_cube(cube)

    nf, nx = cube.shape
    """ Choosing the number of components and initialising the matrix of eigenvectors """

    try:
        xMat, s, vh, u = do_svd(standardized_cube, nc, nx)

    except np.linalg.LinAlgError:
        print("SVD did not converge.")
        return

    return xMat, standardized_cube, s, vh, u


def fit_pca(cross_section, P, T, xMat, nc=3, wav_ind=1):
    """
    Fits the PCA to the opacity data.

    Inputs
    -------
        :cross_section: (n_exp x pixels) it's an array.
        :P: pressure grid
        :T: temperature grid
        :nc: (int) number of PCA components to keep.
        :wav_ind: (int) index of wavelength to fit.

    Returns
    -------
        :xMat: (n_exp x nc) PCA components
        :beta: (nc x pixels) PCA coefficients
    """

    _, beta = fit_mlr(cross_section, xMat)
    return beta


def prep_pca(cross_section, wav_ind=-1, nc=2):
    """
    Prepares the opacity data for PCA.

    Inputs
    -------
        :cross_section: (n_exp x pixels) it's an array.
        :wav_ind: (int) index of wavelength to fit.
        :nc: (int) number of PCA components to keep.

    Returns
    -------
        :xMat: (n_exp x nc) PCA components
    """
    single_pres_single_temp = cross_section[:, :, wav_ind]
    if np.all(single_pres_single_temp == single_pres_single_temp[0, 0]):
        raise ValueError(
            "all values are the same at this wavelength index! Try a different one."
        )

    xMat, fStd, s, vh, u = do_pca(single_pres_single_temp, nc)
    return xMat
