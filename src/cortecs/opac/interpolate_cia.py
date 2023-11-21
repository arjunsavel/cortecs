"""
Functions that can be used to interpolate CIA to a higher
resolution and subsequently chunk it up.
Example instructions / workflow:
>>> reference_file = '../opacFe/opacFe.dat'
>>> CIA_file = 'opacCIA.dat'
>>> interpolate_CIA(CIA_file, reference_file)
>>> CIA_file = 'opacCIA_highres.dat'
>>> ref_file_base = '../opacFe/opacFe'
>>> chunk_wavelengths_CIA(CIA_file, ref_file_base)
And that should work!

todo: generalize to PLATON CIA.

author: @arjunsavel
"""
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from cortecs.opac.chunking import *

######################### Pt. 1: Interpolation ####################################


def get_n_species(CIA_file, verbose=False):
    """
    Returns the number of species in a CIA file.
    Inputs
    -------
        :CIA_file: (str) path to CIA file. e.g., 'opacCIA/opacCIA.dat'
    Outputs
    -------
        :n_species: (int) number of species in the CIA file.
    """

    f = open(CIA_file)
    f1 = f.readlines()
    f.close()
    # the third line fshould be the first normal one.

    line = f1[3]
    n_species = len(line.split()[1:])

    if verbose:
        print(f"Number of species in CIA file {CIA_file}: {n_species}")
        if n_species == 8:
            print(
                "Species are likely: H-el, He-H, CH4-CH4, H2-He, H2-CH4, H2-H, H2-H2, CO2-CO2"
            )
        elif n_species == 14:
            print(
                "Species are likely:  H2-H2, H2-He, H2-H, H2-CH4, CH4-Ar, CH4-CH4, CO2-CO2, He-H, N2-CH4, N2-H2, N2-N2, O2-CO2, O2-N2, O2-O2)"
            )
    return n_species


def get_empty_species_dict(CIA_file, verbose=False):
    """
    returns a species dictioanry given a CIA file
    :param CIA_file:
    :param verbose:
    :return:
    """
    n_species = get_n_species(CIA_file, verbose=verbose)
    if n_species == 8:
        # todo: refactor. has to be a cleaner way to do this! infer the columns, etc.
        Hels = []
        HeHs = []
        CH4CH4s = []
        H2Hes = []
        H2CH4s = []
        H2Hs = []
        H2H2s = []
        CO2CO2s = []

        # python > 3.6 has ordered dictionaries!
        species_dict = {
            "Hels": Hels,
            "HeHs": HeHs,
            "CH4CH4s": CH4CH4s,
            "H2Hes": H2Hes,
            "H2CH4s": H2CH4s,
            "H2Hs": H2Hs,
            "H2H2s": H2H2s,
            "CO2CO2s": CO2CO2s,
        }

    elif n_species == 14:
        H2H2s = []
        H2Hes = []
        H2Hs = []
        H2CH4s = []
        CH4Ar = []
        CH4CH4s = []
        CO2CO2s = []
        HeHs = []
        N2CH4s = []
        N2H2s = []
        N2N2s = []
        O2CO2s = []
        O2N2s = []
        O2O2s = []
        species_dict = {
            "H2H2s": H2H2s,
            "H2Hes": H2Hes,
            "H2Hs": H2Hs,
            "H2CH4s": H2CH4s,
            "CH4Ar": CH4Ar,
            "CH4CH4s": CH4CH4s,
            "CO2CO2s": CO2CO2s,
            "HeHs": HeHs,
            "N2CH4s": N2CH4s,
            "N2H2s": N2H2s,
            "N2N2s": N2N2s,
            "O2CO2s": O2CO2s,
            "O2N2s": O2N2s,
            "O2O2s": O2O2s,
        }
    else:
        print("Number of species in CIA file not recognized. Check the file!")
        species_dict = {}
        species_keys = np.arange(n_species)
        for species_key in species_keys:
            species_dict[species_key] = []
    return species_dict


def read_cia(CIA_file, verbose=False):
    """
    Reads in a CIA file.
    Inputs
    ------
        :CIA_file: (str) path to CIA file. e.g., 'opacCIA/opacCIA.dat'
    Outputs
    -------
        :df: (pd.DataFrame) dataframe with the CIA data.
    """

    f = open(CIA_file)
    f1 = f.readlines()
    f.close()

    temperatures = []
    wavelengths = []

    species_dict = get_empty_species_dict(CIA_file, verbose=verbose)

    # read through all lines in the CIA file
    for line in tqdm(f1[1:], desc="Reading CIA file"):
        if not line or line == "\n":
            continue  # don't want it to break!
        if len(line.split(" ")) == 1 and line != "\n":  # this is a new temperature
            temperature = eval(line[:-1])
            continue  # nothing else on this line
        values = [eval(value) for value in line.split(" ")[::3][:-1]]
        temperatures += [temperature]
        wavelengths += [values[0]]

        for species_ind, species in enumerate(species_dict.keys()):
            species_dict[species] += [values[species_ind + 1]]

    # hm. how to do this with different numbers of species?
    df = pd.DataFrame(
        {
            "temp": temperatures,
            "wav": wavelengths,
        }
    )
    for species in species_dict.keys():
        df[species] = species_dict[species]

    return df


def check_temp_grid(df, real_temperature_grid, CIA_file):
    """
    Checks that the temperature grid of the CIA file is the same as the reference file.
    Inputs
    ------
        :df: (pd.DataFrame) dataframe with the CIA data.
        :real_temperature_grid: (np.array) temperature values for which the opacity file had been
                    computed.
    Outputs
    -------
        None
    Side effects
    -------------
        Raises a ValueError if the temperature grid is not the same.
    """

    for temp in real_temperature_grid:
        if temp not in df.temp.unique():
            print(
                f"Temperature {temp} not in CIA file {CIA_file}! Cannot interpolate in temperature yet. Will set these values to 0."
            )
    return


def interpolate_CIA(CIA_file, reference_file, outfile=None):
    """
    Interpolates a CIA file to a higher resolution, using the wavelength grid
    of a reference file. Note: This function assumes that the CIA file has
    Hels, HeHs, CH4CH4s, H2Hes, H2CH4s, H2Hs, H2H2s, and CO2CO2s.

    TODO: add 2d interpolation.

    Inputs
    ------
        :CIA_file: (str) path to CIA file to be interpolated. e.g.,
                    'opacCIA/opacCIA.dat'
        :reference_file: (str) path to opacity file with the wavelength grid of interest. e.g.,
                    'opacFe/opacFe.dat'
    Outputs
    -------
        None
    Side effects
    -------------
        Creates a file with "hires" attached to the end of CIA_file that has been interpolated
        to higher resolution.
    """

    real_wavelength_grid = get_lams(reference_file, progress=True, filetype="opacity")
    # need to put it on the right temperature grid, too!
    real_temperature_grid = get_temp_grid(
        reference_file, progress=True, filetype="opacity"
    )

    df = read_cia(CIA_file)

    check_temp_grid(df, real_temperature_grid, CIA_file)

    species_dict_interped = get_empty_species_dict(CIA_file)
    interped_wavelengths = []
    interped_temps = []

    # perform interpolation
    for unique_temp in tqdm(real_temperature_grid, desc="Interpolating"):
        if unique_temp not in df.temp.unique():
            for species_key in species_dict_interped.keys():
                species_dict_interped[species_key] += list(
                    np.ones_like(real_wavelength_grid) * 0.0
                )

        else:
            sub_df = df[df.temp == unique_temp]

            # convert to lists so that they're neatly nested lists
            for species_key in species_dict_interped.keys():
                species_dict_interped[species_key] += list(
                    np.interp(real_wavelength_grid, sub_df.wav, sub_df[species_key])
                )

        interped_wavelengths += real_wavelength_grid
        interped_temps += list(np.ones_like(real_wavelength_grid) * unique_temp)

    # write to a new file

    new_string = []

    # this is pretty gross below : (
    reference_species = species_dict_interped[list(species_dict_interped.keys())[0]]

    # todo: include the different temperature range on which to interpolate.

    buffer = "   "  # there's a set of spaces between each string!
    for i in tqdm(range(len(reference_species))):
        # the first line gets different treatment!
        if i == 0:
            temp = np.min(
                interped_temps
            )  # add the LOWEST temperature in the temperature grid!
            new_string += ["{:.12e}".format(temp) + "\n"]
        if interped_temps[i] != temp:
            temp = interped_temps[i]
            new_string += ["{:.12e}".format(temp) + "\n"]
        wavelength_string = "{:.12e}".format(interped_wavelengths[i])

        line_string = wavelength_string + buffer

        for species_key in species_dict_interped.keys():
            line_string += (
                "{:.12e}".format(species_dict_interped[species_key][i]) + buffer
            )

        new_string += [line_string + "\n"]

    # todo: check the insert. and can pull wavelength grid.
    temp_string = " ".join(str(temp) for temp in real_temperature_grid) + " \n"
    new_string.insert(0, temp_string)
    if type(outfile) == type(None):
        outfile = CIA_file.split(".dat")[0]
        outfile += "_highres.dat"
    f2 = open(outfile, "w")
    f2.writelines(new_string)
    f2.close()

    return


######################### Pt. 2: Chunking ####################################


def chunk_wavelengths_CIA(file, ref_file_base, numfiles):
    """
    Performs chunking based on the reference file's wavelength chunking.
    Inputs
    -------
        :file: (str) path to CIA file that should be chunked. e.g., opacCIA_highres.dat.
        :ref_file_base: (str) path base to a set of reference files that are already chunked
                        on the desired wavelength grid. e.g., ../opacFe/opacFe
    """

    header = get_header(file)

    # now get chunks

    f = open(file)
    f1 = f.readlines()
    f.close()

    ticker = 0
    file_suffix = 0

    # determine how many wavelengths are in a given chunk
    if "chunk_list.txt" not in os.listdir():
        chunk_list = []
        for i in tqdm(range(numfiles), desc="Putting together chunk list"):
            chunk_list += [get_wav_per_chunk(i, ref_file_base)]

        np.savetxt("chunk_list.txt", chunk_list)

    else:
        chunk_list = np.loadtxt("chunk_list.txt")

    wav_per_chunk = chunk_list[0]  # to start

    ntemps = 0

    for line in tqdm(
        f1, desc="Writing lines to chunk files"
    ):  # read through all lines in the opacity file
        if not line or line == "\n":
            continue  # don't want it to break

        if len(line.split(" ")) == 1:  # this is a new temperature
            temperature = line
            ntemps += 1
            continue  # nothing else on this line
        ticker += 1
        if ticker == wav_per_chunk:
            file_suffix += 1  # start writing to different file
            ticker = 0
            true_file_suffix = (file_suffix) % numfiles
            wav_per_chunk = chunk_list[true_file_suffix]

            write_to_file(temperature, file, file_suffix, numfiles),
        write_to_file(line, file, file_suffix, numfiles)  # just writes line

    true_header = header[0]

    for chunk_file in glob(file[:-4] + "*"):
        if chunk_file != file:
            prepend_line(chunk_file, true_header)
    return


def get_wav_per_chunk(file_suffix, ref_file_base):
    """
    Grabs the number of wavelengths of a given chunk.
    Inputs
    ------
        :file_suffix: (int) number corresponding to the given chunk. e.g., 1.
        :ref_file_base: (str) path base to a set of reference files that are already chunked
                        on the desired wavelength grid. e.g., ../opacFe/opacFe
    Outputs
    --------
        :len_grid: (int) number of wavelengths in chunk.
    """

    file = ref_file_base + str(file_suffix) + ".dat"

    real_wavelength_grid = get_lams(file, filetype="opacity")

    len_grid = len(real_wavelength_grid)

    return len_grid


def prepend_line(file_name, line):
    """
    Insert given string as a new line at the beginning of a file. Used to fix the header within
    the chunked CIA files.
    """

    # define name of temporary dummy file
    dummy_file = file_name + ".bak"
    # open original file in read mode and dummy file in write mode
    with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + "\n")
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)

    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)
