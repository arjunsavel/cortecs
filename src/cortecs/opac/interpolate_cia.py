"""
Functions that can be used to interpolate CIA to a higher
resolution and subsequently chunk it up.
Example instructions / workflow:
>>> reference_file = '../opacFe/opacFe.dat'
>>> CIA_file = 'opacCIA.dat'
>>> interpolate_cia(CIA_file, reference_file)
>>> CIA_file = 'opacCIA_highres.dat'
>>> ref_file_base = '../opacFe/opacFe'
>>> chunk_wavelengths_cia(CIA_file, ref_file_base)
And that should work!

todo: generalize to PLATON CIA.

author: @arjunsavel
"""
import os
from glob import glob

import numpy as np
from tqdm.autonotebook import tqdm

from cortecs.opac.chunking import *
from cortecs.opac.io import *

# Pt. 1: Interpolation


def check_temp_grid(df, real_temperature_grid, cia_file):
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
                f"Temperature {temp} not in CIA file {cia_file}!"
                + f" Cannot interpolate in temperature yet. Will set these values to 0."
            )
    return


def check_wav_grid(reference_opac, df):
    """
    Checks that the wavelength grid of the CIA file is the same as the reference file.
    Inputs
    ------
        :reference_opac: (Opac) Opac object with the wavelength grid of interest.
        :df: (pd.DataFrame) dataframe with the CIA data.
    Outputs
    -------
        None
    Side effects
    -------------
        Raises a ValueError if the wavelength grid is not the same.
    """
    if reference_opac.wl.max() > df.wav.max() or reference_opac.wl.min() < df.wav.min():
        print(
            "Reference file has a larger wavelength grid than CIA file. Will fill with zeros."
        )
    return


def initialize_species_dict(species_dict, wavelength_grid):
    """
    Initializes a dictionary of species to be interpolated.
    Inputs
    ------
        :species_dict: (dict) dictionary of species to be interpolated.
    Outputs
    -------
        None
    Side effects
    -------------
        Modifies species_dict in place.
    """
    for species_key in species_dict.keys():
        species_dict[species_key] += list(np.ones_like(wavelength_grid) * 0.0)
    return species_dict


def add_line_string_species(line_string, species_dict, i, buffer):
    """
    Adds the species to the line string.
    Inputs
    ------
        :line_string: (str) the line string to be modified.
        :species_dict: (dict) dictionary of species to be interpolated.
        :i: (int) index of the line string.
        :buffer: (str) buffer between each species.
    Outputs
    -------
        None
    Side effects
    -------------
        Modifies line_string in place.
    """
    for species_key in species_dict.keys():
        line_string += "{:.12e}".format(species_dict[species_key][i]) + buffer
    return line_string


def build_new_string(
    interped_wavelengths, interped_temps, reference_species, species_dict_interped
):
    """
    Builds the new string that will be written to the new CIA file.
    Inputs
    ------
        :interped_wavelengths: (list) list of wavelengths that have been interpolated.
        :interped_temps: (list) list of temperatures that have been interpolated.
        :reference_species: (list) list of species that have been interpolated.
        :buffer: (str) buffer between each species.
    Outputs
    -------
        :new_string: (list) list of strings that will be written to the new CIA file.
    Side effects
    -------------
        None
    """
    new_string = []
    buffer = "   "  # there's a set of spaces between each string!
    temp = np.nan  # fill value

    for i in tqdm(range(len(reference_species))):
        # the first line gets different treatment!
        if i == 0:
            temp = np.min(
                interped_temps
            )  # add the LOWEST temperature in the temperature grid!
            new_string += ["{:.12e}".format(temp) + "\n"]
        elif interped_temps[i] != temp:
            temp = interped_temps[i]
            new_string += ["{:.12e}".format(temp) + "\n"]
        wavelength_string = "{:.12e}".format(interped_wavelengths[i])

        line_string = wavelength_string + buffer

        line_string = add_line_string_species(
            line_string, species_dict_interped, i, buffer
        )

        new_string += [line_string + "\n"]

    return new_string


def interpolate_cia(
    cia_file, reference_file, outfile=None, loader="exotransmit", load_kwargs=None
):
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
    if load_kwargs is None:
        load_kwargs = {}
    reference_opac = Opac(reference_file, loader=loader, load_kwargs=load_kwargs)

    real_wavelength_grid = reference_opac.wl

    # need to put it on the right temperature grid, too!
    real_temperature_grid = reference_opac.T

    df = Opac_cia(cia_file, loader="exotransmit_cia", view="full_frame").cross_section

    check_temp_grid(df, real_temperature_grid, cia_file)

    check_wav_grid(reference_opac, df)

    species_dict_interped = get_empty_species_dict(cia_file)
    interped_wavelengths = []
    interped_temps = []

    # perform interpolation. todo: is this redudnant?
    for unique_temp in tqdm(real_temperature_grid, desc="Interpolating"):
        if unique_temp not in df.temp.unique():
            species_dict_interped = initialize_species_dict(
                species_dict_interped, real_wavelength_grid
            )

        else:
            sub_df = df[df.temp == unique_temp]

            # convert to lists so that they're neatly nested lists
            for species_key in species_dict_interped.keys():
                species_dict_interped[species_key] += list(
                    np.interp(real_wavelength_grid, sub_df.wav, sub_df[species_key])
                )

        interped_wavelengths += list(real_wavelength_grid)
        interped_temps += list(np.ones_like(real_wavelength_grid) * unique_temp)

    # write to a new file
    # this is pretty gross below : (
    reference_species = species_dict_interped[list(species_dict_interped.keys())[0]]

    new_string = build_new_string(
        interped_wavelengths, interped_temps, reference_species, species_dict_interped
    )

    # todo: check the insert. and can pull wavelength grid.
    temp_string = " ".join(str(temp) for temp in real_temperature_grid) + " \n"
    new_string.insert(0, temp_string)
    if isinstance(outfile, type(None)):
        outfile = cia_file.split(".dat")[0]
        outfile += "_highres.dat"
    f2 = open(outfile, "w")
    f2.writelines(new_string)
    f2.close()

    return


# Pt. 2: Chunking


def chunk_wavelengths_cia(file, ref_file_base, numfiles):
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

    temperature = np.nan  # fill value
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
            true_file_suffix = file_suffix % numfiles
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

    Inputs
    -------
        :file_name: (str) path to file to be modified.
        :line: (str) line to be inserted.

    Outputs
    --------
        None

    Side effects
    ------------
        Modifies the file in place.
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
