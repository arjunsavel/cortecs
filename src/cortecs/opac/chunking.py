"""
A few functions that can be used to chunk opacity files. Assumes that
opacity files are in the RT code format, that each opacity file to
be chunked is in its on folder, and that we are currently in that folder.


Example instructions / workflow:

>>> file = 'opacTiO.dat'
>>> chunk_wavelengths(file, wav_per_chunk=2598)
>>> filename = 'opacTiO'
>>> add_overlap(filename)

And that should work!

There's some replicated (and likely unnecessary) code, but it hopefully shouldn't
be too confusing. Furthermore, these functions have not been subjected to robust
unit testing, so they might not work right out of the box. Please let me know if
that's the case, and I'd be happy to debug :)

author: @arjunsavel

so far, only works for exo-transmit chunks. TODO: generalize for others.
"""

import os

import numpy as np
from tqdm import tqdm


def chunk_wavelengths(file, nchunks=None, wav_per_chunk=None, adjust_wavelengths=False):
    """
    Performs wavelength-chunking.

    Inputs
    -------
        :file: path to file to be chunked. e.g., 'opacFe.dat'.

        :nchunks: (int or None) Number of chunks to use, splitting the opacity file
                        into roughly even chunks. If None, wav_per_chunk must be specified.

        :wav_per_chunk: (int or None) Number of wavelengths per chunk. If None,
                        nchunks must be specified.

        :adjust_wavelengths: (bool) whether or not to scale from CGS to MKS. For most situations,
                        can be kept false.

    Outputs
    -------
        None

    Side effects
    -------------
        Creates a number of files in same directory as file, titled file*.dat.
    """
    if nchunks and wav_per_chunk:
        print(
            "Both nchunks and wav_per_chunk specified. Will default to specified wav_per_chunk."
        )
    if not nchunks and not wav_per_chunk:
        raise ValueError(
            "Cannot set nchunks and wav_per_chunk to None! One must be specified."
        )

    if not wav_per_chunk:
        num_wavelengths = count_wavelengths(file)
        wav_per_chunk = round(num_wavelengths / nchunks)

    header = get_header(file)

    # now get chunks

    f = open(file)
    f1 = f.readlines()

    ticker = 0
    file_suffix = 0

    # read through all lines in the opacity file. todo: from already read in opacity?
    for x in tqdm(f1):
        if not x:
            continue
        commad = x.replace(" ", ",")
        try:
            if len(np.array([eval(commad)]).flatten()) == 1:  # if a wavelength line
                ticker += 1
            elif (
                len(x.split(" ")) == 48 and adjust_wavelengths
            ):  # this is ntemp, I believe
                x = adjust_wavelength_unit(x, 1e-4, style="full")
        #                pass # don't need to adust wavelengths anymore!
        except:
            pdb.set_trace()

        if ticker == wav_per_chunk:
            file_suffix += 1  # start writing to different file
            ticker = 0
            write_to_file(header, file, file_suffix)
        write_to_file(x, file, file_suffix)

    f.close()
    return


def write_to_file(line, file, file_suffix):
    """
    Writes (appends, really) a line to a file.

    Inputs:
        :line: (str) line to write to file.
        :file: (str) base path to file to be written to. e.g., 'opacFe'
        :file_suffix: (int) suffix to add, usually chunk number. e.g. 5

    Outputs:
        None

    Side effects:
        Writes a line to a file!
    """
    true_filename = f"{file[:-4] + str(file_suffix) + '.dat'}"
    f = open(true_filename, "a")
    f.write(line)
    f.close()


def get_header(file):
    """
    Gets the header of a file.

    Inputs:
        :file: (str) path to file whose header we want.

    Outputs:
        header of file (string)
    """

    f = open(file)
    f1 = f.readlines()
    f.close()

    return f1[0] + f1[1]


def count_wavelengths(file):
    """
    parses through wavelength file to see how many wavelength points it has.
    todo: can just get from the counting wavelengths?

        Inputs
        ------
                :file: (str) path to file to be parsed
    Outputs
    --------
        :ticker: (int) number of wavelength points in the file.
    """
    ticker = 0
    f = open(file)
    f1 = f.readlines()
    ticker = 0
    f.close()
    for x in tqdm(f1):
        commad = x.replace(" ", ",")
        #     print(eval(commad)) # don't actually use this -- floating point error!
        if len(np.array([eval(commad)]).flatten()) == 1:  # aha! here's the marker.
            ticker += 1

    return ticker


def get_lams(file):
    """
    Takes in an opacity file and returns an array of all wavelengths within the file.

    Inputs:
        :file: (str) path to opacity file.

    Outputs:
        :wavelengths: (numpy.array) individual wavelength points within the opacity file [m]

    todo: refactor so this isn't largely replicated in io.
    """
    f = open(file)
    f1 = f.readlines()

    ticker = 0
    wavelengths = []

    # read through all lines in the opacity file
    for x in f1:
        # check if blank line
        if not x:
            continue

        # check if a wavelength line
        commad = x.replace(" ", ",")
        if len(np.array([eval(commad)]).flatten()) == 1:
            wavelengths += [eval(x[:-1])]
    f.close()
    return np.array(wavelengths)


def add_lams(max_lam_to_add_ind, file, next_file):
    """
    Takes the first `max_lam_to_add_ind` opacities from next_file and appends them to file.

    Inputs:
        :max_lam_to_add_ind: (int) number of wavelength points to add from one file to the other.
        :file: (str) (str) path to file to which wavelength points are being *added*.
        :next_file: (str) path to file from which wavelength points are being drawn.

    Outputs:
        None

    Side effects:
        Modifies file.
    """
    try:
        f = open(next_file)
    except FileNotFoundError:
        print(f"{next_file} not found. Moving on!")
        return
    f1 = f.readlines()[2:]  # first two files of opacity are header info
    f.close()
    if max_lam_to_add_ind >= len(f1):
        raise IndexError("Try choosing a larger chunk size.")

    ticker = 0

    # read through all lines in the opacity file
    for x in f1:
        # skip blank lines
        if not x:
            continue

        # check if wavelength line
        commad = x.replace(" ", ",")
        if len(np.array([eval(commad)]).flatten()) == 1:
            ticker += 1

        # append to file
        f2 = open(file, "a")
        f2.write(x)
        f2.close()
        if ticker == max_lam_to_add_ind:
            return


def add_previous(num_to_add, file, previous_file):
    """
    Adds a certain number of wavelength points to a file from a previous one.

    Inputs:
        :num_to_add: (int) number of wavelength points to add from one file to the other.
        :file: (str) (str) path to file to which wavelength points are being *added*.
        :previous_file: (str) path to file from which wavelength points are being drawn.

    Outputs:
        None

    Side effects:
        Modifies file.
    """
    try:
        f = open(previous_file)
    except FileNotFoundError:
        print(f"{previous_file} not found. Moving on!")
        return

    f1 = f.readlines()[2:]  # first two files of opacity are header info
    f.close()

    ticker = 0

    # read through all lines in the opacity file
    for x in f1[::-1]:
        if not x:
            continue

        commad = x.replace(" ", ",")
        if len(np.array([eval(commad)]).flatten()) == 1:  # if a wavelength line
            ticker += 1

        # append line to file
        f2 = open(file, "a")
        f2.write(x)
        f2.close()
        if ticker == num_to_add:
            return


def add_overlap(filename, v_max=11463.5):
    """
    Adds overlap from file n+1 to file n. The last file has nothing added to it. This
    step is necessary for the Doppler-on version of the RT code.

    ***Assumes that the current directory only has files labeled 'filename*.dat'***

    Inputs:
        :filename: (str) "base name" of the opacity chunks. e.g., 'opacFe', corresponding to
                        'opacFe*.dat'.

        :v_max: (float, default 11463.5) maximum velocity that will be Doppler-shifted to in the
                        RT code. Twice this value is used to calculate how much overlap to include
                        (+ an extra 20 wavelength points, to be safe!). The default comes from Tad's
                        WASP-76b GCM outputs. [m/s]


    Output:
        None

    Side effects:
        Modifies every 'filename*.dat' file.
    """
    for i in tqdm(
        range(len(os.listdir()[:-1])), position=0, leave=True
    ):  # don't include the last file
        file = filename + str(i) + ".dat"

        next_file = filename + str(i + 1) + ".dat"

        # go into file n to determine the delta lambda

        curr_lams = get_lams(file)

        try:
            next_lams = get_lams(next_file)
        except FileNotFoundError:
            print(f"{next_file} not found. Moving on!")
            continue

        c = 3e8  # m/s
        max_curr_lam = np.max(curr_lams)
        delta_lam = 2 * max_curr_lam * v_max / c  # delta_lambda/lambda = v/c

        # add another 20 indices to be safe!
        max_lam_to_add_ind = (
            np.argmin(np.abs(next_lams - (max_curr_lam + delta_lam))) + 20
        )

        add_lams(max_lam_to_add_ind, file, next_file)


if __name__ == "__main__":
    pass
    # example workflow below!
    # os.chdir('/libra1/asavel/wasp_76b_opacities')
    # species_list = [  # 'O',
    #     # 'Mn_plus',
    #     # 'Si',
    #     # 'Cu',
    #     # 'Cr_plus',
    #     # 'V_plus',
    #     # 'Ga',
    #     # 'Rb',
    #     # 'Sr_plus',
    #     # 'Sr',
    #     # 'Y_plus',
    #     # 'Y',
    #     # 'Sc',
    #     'Sc_plus',
    #     'Al',
    #     'Co']
    # for species in tqdm(species_list):
    #     os.chdir('opac' + species)
    #     file = f'opac{species}.dat'
    #     chunk_wavelengths(file, wav_per_chunk=2598)
    #     filename = file.split('.')[0]
    #     add_overlap(filename)
    #     os.chdir('..')
