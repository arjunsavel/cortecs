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


def interpolate_CIA(CIA_file, reference_file):
    """
    Interpolates a CIA file to a higher resolution, using the wavelength grid
    of a reference file. Note: This function assumes that the CIA file has
    Hels, HeHs, CH4CH4s, H2Hes, H2CH4s, H2Hs, H2H2s, and CO2CO2s.
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

    real_wavelength_grid = get_wav_grid(reference_file, progress=True)

    f = open(CIA_file)
    f1 = f.readlines()
    f.close()

    temperatures = []
    wavelengths = []
    # todo: refactor. has to be a cleaner way to do this! infer the columns, etc.
    Hels = []
    HeHs = []
    CH4CH4s = []
    H2Hes = []
    H2CH4s = []
    H2Hs = []
    H2H2s = []
    CO2CO2s = []

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
        Hels += [values[1]]
        HeHs += [values[2]]
        CH4CH4s += [values[3]]
        H2Hes += [values[4]]
        H2CH4s += [values[5]]
        H2Hs += [values[6]]
        H2H2s += [values[7]]
        CO2CO2s += [values[8]]

    # easier to slice with a dataframe later on
    df = pd.DataFrame(
        {
            "temp": temperatures,
            "wav": wavelengths,
            "Hel": Hels,
            "HeH": HeHs,
            "CH4CH4": CH4CH4s,
            "H2He": H2Hes,
            "H2CH4": H2CH4s,
            "H2H": H2Hs,
            "H2H2": H2H2s,
            "CO2CO2": CO2CO2s,
        }
    )

    # perform interpolation

    interped_Hels = []
    interped_HeHs = []
    interped_CH4CH4s = []
    interped_H2Hes = []
    interped_H2CH4s = []
    interped_H2Hs = []
    interped_H2H2s = []
    interped_CO2CO2s = []
    interped_wavelengths = []
    interped_temps = []

    for unique_temp in tqdm(df.temp.unique()):
        sub_df = df[df.temp == unique_temp]

        # convert to lists so that they're neatly nested lists
        interped_Hels += list(np.interp(real_wavelength_grid, sub_df.wav, sub_df.Hel))
        interped_HeHs += list(np.interp(real_wavelength_grid, sub_df.wav, sub_df.HeH))
        interped_CH4CH4s += list(
            np.interp(real_wavelength_grid, sub_df.wav, sub_df.CH4CH4)
        )
        interped_H2Hes += list(np.interp(real_wavelength_grid, sub_df.wav, sub_df.H2He))
        interped_H2CH4s += list(
            np.interp(real_wavelength_grid, sub_df.wav, sub_df.H2CH4)
        )
        interped_H2Hs += list(np.interp(real_wavelength_grid, sub_df.wav, sub_df.H2H))
        interped_H2H2s += list(np.interp(real_wavelength_grid, sub_df.wav, sub_df.H2H2))
        interped_CO2CO2s += list(
            np.interp(real_wavelength_grid, sub_df.wav, sub_df.CO2CO2)
        )

        interped_wavelengths += real_wavelength_grid
        interped_temps += list(np.ones_like(real_wavelength_grid) * unique_temp)

    # write to a new file

    new_string = []

    buffer = "   "  # there's a set of spaces between each string!
    for i in tqdm(range(len(interped_Hels))):
        # the first line gets different treatment!
        if i == 0:
            temp = 500
            new_string += ["{:.12e}".format(temp) + "\n"]
        if interped_temps[i] != temp:
            temp = interped_temps[i]
            new_string += ["{:.12e}".format(temp) + "\n"]
        wavelength = "{:.12e}".format(interped_wavelengths[i])
        Hel_string = "{:.12e}".format(interped_Hels[i])
        HeH_string = "{:.12e}".format(interped_HeHs[i])
        CH4CH4_string = "{:.12e}".format(interped_CH4CH4s[i])
        H2He_string = "{:.12e}".format(interped_H2Hes[i])
        H2CH4_string = "{:.12e}".format(interped_H2CH4s[i])
        H2H_string = "{:.12e}".format(interped_H2Hs[i])
        H2H2_string = "{:.12e}".format(interped_H2H2s[i])
        CO2CO2_string = "{:.12e}".format(interped_CO2CO2s[i])
        new_string += [
            wavelength
            + buffer
            + Hel_string
            + buffer
            + HeH_string
            + buffer
            + CH4CH4_string
            + buffer
            + H2He_string
            + buffer
            + H2CH4_string
            + buffer
            + H2H_string
            + buffer
            + H2H2_string
            + buffer
            + CO2CO2_string
            + buffer
            + "\n"
        ]

    # todo: check the insert. and can pull wavelength grid.
    new_string.insert(
        0,
        "500.000 600.000 700.000 800.000 900.000 1000.000 1100.000 1200.000 1300.000 1400.000 1500.000 1600.000 1700.000 1800.000 1900.000 2000.000 2100.000 2200.000 2300.000 2400.000 2500.000 2600.000 2700.000 2800.000 2900.000 3000.000 3100.000 3200.000 3300.000 3400.000 3500.000 3600.000 3700.000 3800.000 3900.000 4000.000 4100.000 4200.000 4300.000 4400.000 4500.000 4600.000 4700.000 4800.000 4900.000 5000.000 \n",
    )
    new_file = CIA_file.split(".dat")[0]
    new_file += "_highres.dat"
    f2 = open(new_file, "w")
    f2.writelines(new_string)
    f2.close()

    return


def get_wav_grid(file, progress=False):
    """
    Returns the wavelength grid used in an opacity file.
    Inputs
    -------
        :file: (str) path to opacity file with the wavelength grid of interest. e.g.,
                    'opacFe/opacFe.dat'
        :progress: (bool) whether or not to include a progress bar (if tqdm is installed).
                    Useful for the biggest opacity file!
    Outputs
    -------
        :wav_grid: (np.array) wavelength values for which the opacity file had been
                    computed.
    """

    wav_grid = []

    f = open(file)
    f1 = f.readlines()
    f.close()

    if progress:
        iterator = tqdm(f1[2:], desc="Grabbing wavelength grid")
    else:
        iterator = f1[2:]

    # read through all lines in the opacity file; first few lines are header!
    for x in iterator:
        # skip blank lines
        if not x:
            continue
        commad = x.replace(" ", ",")

        # check if a wavelength line
        if len(np.array([eval(commad)]).flatten()) == 1:
            wav_grid += [np.array([eval(commad)]).flatten()[0]]

    return wav_grid


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

            write_to_file(temperature, file, file_suffix, ntemps, numfiles)
        write_to_file(line, file, file_suffix, ntemps, numfiles)  # just writes line

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

    real_wavelength_grid = get_wav_grid(file)

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


def write_to_file(line, file, file_suffix, ntemps, numfiles):
    """
    Writes a line to a file.
    Inputs
    ------
        :line: (str) line to be written.
        :file: (str) path to file being chunked. e.g., opacCIA_highres.dat
        :file_suffix: (int) number of chunk.
        :ntemps: (int) number of temperatures in grid.
    Outputs
    --------
        None
    Side effects
    -------------
        Writes a line to a file!
    """
    true_file_suffix = (file_suffix) % numfiles
    true_filename = f"{file[:-4] + str(true_file_suffix) + '.dat'}"
    f = open(true_filename, "a")
    f.write(line)
    f.close()
