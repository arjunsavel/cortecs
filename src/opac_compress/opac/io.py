"""
Reads opacity data from various sources.



author: @arjunsavel
"""
import h5py
import numpy as np

class loader_base(object):
    """
    loads in opacity data from various sources. To be passed on to Opac object.
    """

    def __init__(self):
        """
        nothing to do here
        """
        pass

    def load(self, filename, loader='chimera'):
        """
        loads in opacity data from various sources. To be passed on to Opac object.
        """

        hf = h5py.File(filename, 'r')
        wno = np.array(hf['wno'], dtype=np.float64)
        T = np.array(hf['T'], dtype=np.float64)
        P = np.array(hf['P'], dtype=np.float64)
        cross_section = np.array(hf['xsec'], dtype=np.float64)
        hf.close()
        wl = 1e4/wno
        return wl, T, P, cross_section

class loader_heliosk(loader_base):
    """
    loads in opacity data from the heliosk database.
    """

    def load(self, f):
