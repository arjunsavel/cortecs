"""
Reads opacity data from various sources.

author: @arjunsavel
"""

import pickle

import h5py
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

AMU = 1.6605390666e-24  # atomic mass unit in cgs. From astropy!


class loader_base(object):
    """
    loads in opacity data from various sources. To be passed on to Opac object.

    todo: tutorial on how to use this?
    """

    def __init__(
        self,
        wl_key="wno",
        T_key="T",
        P_key="P",
        cross_section_key="xsec",
        wl_style="wno",
        temperature_axis=0,
        pressure_axis=1,
        wavelength_axis=2,
    ):
        """
        sets the keys for the loader object.
        """
        self.wl_key = wl_key
        self.T_key = T_key
        self.P_key = P_key
        self.cross_section_key = cross_section_key
        self.wl_style = wl_style
        self.temperature_axis = temperature_axis
        self.pressure_axis = pressure_axis
        self.wavelength_axis = wavelength_axis

    def load(self, filename):
        """
        loads in opacity data from various sources. To be passed on to Opac object.

        Inputs
        ------
        filename : str
            name of file to load

        Outputs
        -------
        wl : np.ndarray
            wavelengths
        T : np.ndarray
            temperatures
        P : np.ndarray
            pressures
        cross_section : np.ndarray
            cross sections for the opacity
        """

        hf = h5py.File(filename, "r")
        wl = np.array(hf[self.wl_key], dtype=np.float64)
        T = np.array(hf[self.T_key], dtype=np.float64)
        P = np.array(hf[self.P_key], dtype=np.float64)
        cross_section = np.array(hf[self.cross_section_key], dtype=np.float64)
        hf.close()

        # want temperature index 0, pressure to 1, wavelength to 2 for standard usage.
        cross_section = np.moveaxis(
            cross_section,
            [self.temperature_axis, self.pressure_axis, self.wavelength_axis],
            [0, 1, 2],
        )

        if self.wl_style == "wno":
            wl = 1e4 / wl
            wl *= 1e-6  # now in meters

        if np.all(np.diff(wl)) <= 0:
            wl = wl[::-1]
            cross_section = cross_section[:, :, ::-1]
            # reverse!
        return wl, T, P, cross_section


class loader_chimera(loader_base):
    """
    loads in opacity data that are produced with the CHIMERA code.
    """


class loader_helios(loader_base):
    """
    loads in opacity data that are produced with the HELIOS ktable function.
    """

    wl_key = "wavelengths"
    T_key = "temperatures"
    P_key = "pressures"
    cross_section_key = "opacities"
    wl_style = "wl"


class loader_platon(loader_base):
    """
    loads in opacity data that are used with the PLATON code.
    """

    # atomic weights of molecules
    species_weight_dict = {
        "CO": 28.01,
        "H2O": 18.015,
        "CH4": 16.04,
        "NH3": 17.031,
        "CO2": 44.009,
        "HCN": 27.026,
        "C2H2": 26.038,
        "H2S": 34.08,
        "PH3": 33.997,
        "VO": 66.94,
        "TiO": 63.866,
        "Na": 22.99,
        "K": 39.098,
        "FeH": 55.845,
        "H2": 2.016,
        "He": 4.003,
        "H-": 1.008,
        "H": 1.008,
        "He+": 4.003,
        "H+": 1.008,
        "e-": 0.00054858,
        "H2+": 2.016,
        "H2-": 2.016,
        "H3+": 3.024,
        "H3-": 3.024,
        "H2O+": 18.015,
        "H2O-": 18.015,
        "CH4+": 16.04,
        "CH4-": 16.04,
        "C2H4": 28.054,
    }

    def load(self, cross_sec_filename, T_filename="", P_filename="", wl_filename=""):
        """
        loads in opacity data that's built for PLATON. To be passed on to Opac object.

        The temperature grid, pressure grid, and wavelength grid are saved as separate files for PLATON.

        Inputs
        ------
        filename : str
            name of file to load
        cross_sec_filename : str
            name of cross section file
        T_filename : str
            name of temperature file
        P_filename : str
            name of pressure file
        wl_filename : str
            name of wavelength file
        """
        # todo: check wl units. They're in meters here.
        # temperatures are in K.
        # pressures are in Pa, I believe.
        wl = np.load(wl_filename, allow_pickle=True)
        T = np.load(T_filename, allow_pickle=True)
        P = np.load(P_filename, allow_pickle=True)
        cross_section = np.load(
            cross_sec_filename, allow_pickle=True
        )  # packaged as T x P x wl. todo: check packing
        # cross-section units for fitting...? keep the same internally.
        species = cross_sec_filename.split("_")[-1].split(".")[0]
        try:
            self.species_weight = self.species_weight_dict[species]
        except KeyError:
            raise KeyError(
                f"Species {species} read from filename and not found in species_weight_dict. Please add it."
            )
        cross_section = cross_section * AMU * self.species_weight * 1e-4

        # set a floor to the opacities, as per the floor used in the PLATON code's opacity files.
        cross_section[np.less(cross_section, 1e-104)] = 1e-104

        # and now make it log10
        cross_section = np.log10(cross_section)

        return wl, T, P, cross_section


class loader_platon_cia(loader_base):
    """
    loads in opacity data that are used with the PLATON code's collision-induced absorption..
    """

    def load(self, cross_sec_filename, T_filename, wl_filename, species_name):
        """
        loads in opacity data that's built for PLATON. To be passed on to Opac object.

        The temperature grid, pressure grid, and wavelength grid are saved as separate files for PLATON.

        Inputs
        ------
        filename : str
            name of file to load
        cross_sec_filename : str
            name of cross section file
        T_filename : str
            name of temperature file
        wl_filename : str
            name of wavelength file
        species_name : tuples
            name of two colliding species. E.g., ('H2', 'CH4'). todo: all at once?
        """
        # todo: check wl units. They're in meters here.
        # temperatures are in K.
        # pressures are in Pa, I believe.
        wl = np.load(wl_filename, allow_pickle=True)
        T = np.load(T_filename, allow_pickle=True)
        data = pickle.load(
            open(cross_sec_filename, "rb"), encoding="latin1"
        )  # packaged as T x P x wl. todo: check packing
        cross_section = data[species_name]
        return wl, T, cross_section


class loader_exotransmit_cia(loader_base):
    """
    loads in opacity data that are used with the PLATON code's collision-induced absorption..
    """

    def check_line_break(self, line, temperature):
        """
        Checks whether the given line should be skipped

        """
        truth_val = False
        if not line or line == "\n":
            truth_val = True
        if len(line.split()) == 1 and line != "\n":  # this is a new temperature
            temperature = eval(line[:-1])
            truth_val = True
        return truth_val, temperature

    def load(self, cross_sec_filename, verbose=False):
        """
        loads in opacity data that's built for exo-transmit. To be passed on to Opac object.


        Inputs
        ------
            :CIA_file: (str) path to CIA file. e.g., 'opacCIA/opacCIA.dat'
        Outputs
        -------
            :df: (pd.DataFrame) dataframe with the CIA data.
        """

        f = open(cross_sec_filename)
        f1 = f.readlines()
        f.close()

        temperatures = []
        wavelengths = []

        species_dict = get_empty_species_dict(cross_sec_filename, verbose=verbose)

        # read through all lines in the CIA file
        temperature = 0.0  # initialize

        for line in tqdm(f1[1:], desc="Reading CIA file"):
            truth_val, temperature = self.check_line_break(line, temperature)

            if truth_val:
                continue

            values = [eval(value) for value in line.split()]
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

        columns = list(df.columns)
        columns.remove("temp")
        columns.remove("wav")

        return df.wav.values, df.temp.values, df[columns]


class loader_exotransmit(loader_base):
    """
    loads in opacity data in the exo-transmit format. This format is also used by PLATON, I believe.
    """

    def get_t_p(self, file):
        """
        Gets the temperatures and pressures of a file from its header.

        Inputs:
            :file: (str) path to file whose header we want.

        Outputs:
            header of file (string)
        """

        f = open(file)
        f1 = f.readlines()
        f.close()

        t_line = f1[0]
        p_line = f1[1]

        temperature_strings = t_line.split(" ")[:-1]
        T = np.array([eval(temp) for temp in temperature_strings])

        pressure_strings = p_line.split(" ")[:-1]
        P = np.array([eval(pres) for pres in pressure_strings])

        del f1
        return T, P

    def get_lams_and_opacities(self, file):
        """
        Takes in an opacity file and returns an array of all wavelengths within the file.

        Returns the opacities, as well — so that the file is only iterated through once.

        Inputs:
            :file: (str) path to opacity file.

        Outputs:
            :wavelengths: (numpy.array) individual wavelength points within the opacity file [m]
        """
        f = open(file)
        f1 = f.readlines()

        wavelengths = []
        opacities = []

        # read through all lines in the opacity file
        # skip through the header!
        for x in tqdm(f1[2:], desc="reading exotransmit wavelengths"):
            # # check if blank line
            if not x:
                continue
            # check if a wavelength line
            # commad = x.replace(" ", ",")
            # if len(np.array([eval(commad)]).flatten()) == 1:
            #     wavelengths.append(eval(x[:-1]))
            # else:
            #     # the first entry in each opacity line is the pressure
            #     opacity_string = x.split()[1:]
            #     opacity_vals = [eval(opacity) for opacity in opacity_string]
            #     opacities.append(opacity_vals)
            if not x.strip():
                continue
                # check if a wavelength line
            line_values = x.split()
            if len(line_values) == 1:
                wavelengths.append(float(line_values[0]))
            else:
                # the first entry in each opacity line is the pressure
                opacity_vals = [float(opacity) for opacity in line_values[1:]]
                opacities.append(opacity_vals)

        f.close()
        # pdb.set_trace()
        # oh this isn't reshaped haha
        del f1
        try:
            return np.array(wavelengths), np.array(opacities)
        except:
            pdb.set_trace()

    def load(self, filename, fullfile=True):
        """
        Loads file.

        Inputs
        ------
        filename : str
            name of file to load

        Outputs
        -------
        wl : np.ndarray
            wavelengths
        T : np.ndarray
            temperatures
        P : np.ndarray
            pressures
        """

        wl, cross_section = self.get_lams_and_opacities(filename)
        T, P = self.get_t_p(filename)
        P /= 1e5  # internally in bars

        cross_section[np.less(cross_section, 1e-104)] = 1e-104

        # and now make it log10
        cross_section = np.log10(cross_section)

        if fullfile:  # only reshape if it's not a "test" file.
            cross_section = cross_section.reshape(len(T), len(P), len(wl))

        return wl, T, P, cross_section


# todo: put these in the opac class?
def get_empty_species_dict(CIA_file, verbose=False):
    """
    returns a species dictioanry given a CIA file

    Inputs
    -------
        :CIA_file: (str) path to CIA file. e.g., 'opacCIA/opacCIA.dat'
        :verbose: (bool) whether to print out the species that are likely in the file.

    Outputs
    -------
        :species_dict: (dict) dictionary of species.

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
    del f1
    return n_species


# todo: write a writer for each of these.


class writer_base(object):
    def __init__(
        self,
        wl_key="wno",
        T_key="T",
        P_key="P",
        cross_section_key="xsec",
        wl_style="wno",
    ):
        """
        nothing to do here
        """
        self.wl_key = wl_key
        self.T_key = T_key
        self.P_key = P_key
        self.cross_section_key = cross_section_key
        self.wl_style = wl_style
        pass


class writer_exotransmit_cia(writer_base):
    """
    does writing for the CIA object, takng in an opac object.
    """

    def write(self, opac, outfile, verbose=False):
        """
        loads in opacity data that's built for exo-transmit. To be passed on to Opac object.


        Inputs
        ------
            :CIA_file: (str) path to CIA file. e.g., 'opacCIA/opacCIA.dat'
        Outputs
        -------
            :df: (pd.DataFrame) dataframe with the CIA data.
        """

        new_string = []
        print("optimized time")
        # don't want to write temp and wav
        columns = list(opac.cross_section.columns)
        if "temp" in columns:
            columns.remove("temp")
        if "wav" in columns:
            columns.remove("wav")

        self.species_dict_interped = opac.cross_section[columns].to_dict()
        self.interped_temps = opac.T
        self.interped_wavelengths = opac.wl

        # this is pretty gross below : (
        reference_species = self.species_dict_interped[
            list(self.species_dict_interped.keys())[0]
        ]

        # todo: include the different temperature range on which to interpolate.

        self.buffer = "   "  # there's a set of spaces between each string!
        temp = 0.0  # initial value
        for i in tqdm(range(len(reference_species)), desc="Writing file"):
            new_string, temp = self.append_line_string(
                new_string,
                i,
                temp,
            )

        # todo: check the insert. and can pull wavelength grid.
        temp_string = (
            " ".join(str(temp) for temp in np.sort(np.unique(self.interped_temps)))
            + " \n"
        )
        new_string.insert(0, temp_string)

        f2 = open(outfile, "w")
        f2.writelines(new_string)
        f2.close()

    def append_line_string(
        self,
        new_string,
        i,
        temp,
    ):
        # the first line gets different treatment!
        # if i == 0:
        #    temp = np.min(
        #        self.interped_temps
        #    )  # add the LOWEST temperature in the temperature grid!
        #    new_string += ["{:.12e}".format(temp) + "\n"]
        # if self.interped_temps[i] != temp:
        #    temp = self.interped_temps[i]
        #    new_string += ["{:.12e}".format(temp) + "\n"]
        # wavelength_string = "{:.12e}".format(self.interped_wavelengths[i])######

        # line_string = wavelength_string + self.buffer

        # for species_key in self.species_dict_interped.keys():
        # again, this works because python dicts are ordered in 3.6+
        #    line_string += (
        #        "{:.12e}".format(
        #            list(self.species_dict_interped[species_key].values())[i]
        #        )
        #        + self.buffer
        #    )

        # new_string += [line_string + "\n"]

        # return new_string, temp

        if i == 0:
            temp = np.min(self.interped_temps)
            new_string.append("{:.12e}\n".format(temp))
        if self.interped_temps[i] != temp:
            temp = self.interped_temps[i]
            new_string.append("{:.12e}\n".format(temp))

        wavelength_string = "{:.12e}".format(self.interped_wavelengths[i])
        line_string = wavelength_string + self.buffer

        species_values = [
            species_dict_value[i]
            for species_dict_value in self.species_dict_interped.values()
        ]
        line_string += self.buffer.join(
            "{:.12e}".format(value) for value in species_values
        )

        new_string.append(line_string + "\n")
        return new_string, temp
        # todo: maybe the loader objects should also take an opac object. for parallel structure : )


class writer_chimera(writer_base):
    """
    does writing for the exotransmit object, takng in an opac object.
    """

    def write(self, opac, outfile, verbose=False):
        """
        loads in opacity data that's built for exo-transmit. To be passed on to Opac object.
        """

        # below is the write
        wl = 1e6 * opac.wl  # now in microns
        wno = 1e4 / wl  # now in cm^-1
        wno = wno[::-1]
        cross_section = opac.cross_section[:, ::-1]
        cross_section = np.moveaxis(cross_section, 0, -1)
        cross_section = np.moveaxis(cross_section, 0, -1)
        cross_section = np.moveaxis(cross_section, 0, -1)  # this
        # want temperature index 0, pressure to 1, wavelength to 2 for standard usage.

        hf = h5py.File(outfile, "w")
        hf.create_dataset(self.wl_key, data=wno)
        hf.create_dataset(self.T_key, data=opac.T)
        hf.create_dataset(self.P_key, data=opac.P)
        hf.create_dataset(self.cross_section_key, data=cross_section)

        # ah man darn the xsec shape is wrong.

        hf.close()

        # todo: once this is done, check that everything is in the correct units.
        return
