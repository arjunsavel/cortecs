"""
Reads opacity data from various sources.



author: @arjunsavel
"""
import h5py
import numpy as np
from tqdm import tqdm

class loader_base(object):
    """
    loads in opacity data from various sources. To be passed on to Opac object.
    """

    wl_key = 'wno'
    T_key = 'T'
    P_key = 'P'
    cross_section_key = 'xsec'
    wl_style = 'wno'

    def __init__(self):
        """
        nothing to do here
        """
        pass

    def load(self, filename):
        """
        loads in opacity data from various sources. To be passed on to Opac object.
        """

        hf = h5py.File(filename, 'r')
        wl = np.array(hf[self.wl_key], dtype=np.float64)
        T = np.array(hf[self.T_key], dtype=np.float64)
        P = np.array(hf[self.P_key], dtype=np.float64)
        cross_section = np.array(hf[self.cross_section_key], dtype=np.float64)
        hf.close()

        if self.wl_style == 'wno':
            wl = 1e4 / wl
        return wl, T, P, cross_section

class loader_helios(loader_base):
    """
    loads in opacity data that are produced with the HELIOS ktable function.
    """
    wl_key = 'wavelengths'
    T_key = 'temperatures'
    P_key = 'pressures'
    cross_section_key = 'opacities'
    wl_style = 'wl'

class loader_platon(loader_base):
    """
    loads in opacity data that are used with the PLATON code.
    """
    # needs the numpy stuff
    raise NotImplementedError


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

        temperature_strings = t_line.split(' ')[:-1]
        T = np.array([eval(temp) for temp in temperature_strings])

        pressure_strings = p_line.split(' ')[:-1]
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
        for x in tqdm(f1, desc='reading wavelengths'):
            # check if blank line
            if not x:
                continue
            # check if a wavelength line
            commad = x.replace(" ", ",")
            if len(np.array([eval(commad)]).flatten()) == 1:
                wavelengths += [eval(x[:-1])]
            else:
                # the first entry in each opacity line is the pressure
                opacity_string = commad.split(' ')[1:]
                opacity_vals = np.array([eval(opacity) for opacity in opacity_string])
                opacities += [opacity_vals]

        f.close()
        del f1
        return np.array(wavelengths), np.array(opacities)

    def load(self, filename):
        wl, opacities = self.get_lams_and_opacities(filename)
        T, P = self.get_t_p(filename)

        return wl, T, P, opacities



