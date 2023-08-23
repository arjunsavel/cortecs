"""
this file holds the class describing opacity data. hmm...maybe I want the loader in a different file?

author: @arjunsavel
"""
from cortecs.io import *

class Opac(object):
    """
    this class holds the opacity data and provides methods for evaluating the opacity at a given temperature and pressure.

    Everything's already been loaded into memory, so this is just a wrapper for the data.
    """
    def __init__(self, filename, loader='chimera'):
        """
        wraps around the loaders.

        Parameters
        ----------
        filename : str
            name of file to load
        loader : str
            name of loader to use. default is chimera.

        Returns
        -------
        nothing
        """
        self.filename = filename
        load_obj = loader_base()
        self.wl, self.T, self.P, self.cross_section = load_obj.load(filename, loader=loader)

        self.n_wav, self. n_t, self.n_p = len(self.wl), len(self.T), len(self.P)

        return