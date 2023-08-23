"""
this file holds the class describing opacity data. hmm...maybe I want the loader in a different file?

author: @arjunsavel
"""
from cortecs.io import *


class Opac(object):
    """
    this class holds the opacity data and provides methods for evaluating the opacity at a given temperature and pressure.

    Everything's already been loaded into memory, so this is just a wrapper for the data.

    todo: use the different loaders
    """

    method_dict = {
        "chimera": loader_base,
        "helios": loader_helios,
        "platon": loader_platon,
        "exotransmit": loader_exotransmit,
    }

    def __init__(self, filename, loader="chimera"):
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
        self.wl, self.T, self.P, self.cross_section = load_obj.load(
            filename, loader=loader
        )

        self.n_wav, self.n_t, self.n_p = len(self.wl), len(self.T), len(self.P)

        return

    def _get_loader(self, loader_name):
        """
        gets the loader object

        Parameters
        ----------
        loader_name : str
            name of loader

        Returns
        -------
        loader object
        """
        if loader_name not in self.method_dict:
            raise ValueError(
                "loader name not found. valid loaders are: {}".format(
                    self.method_dict.keys()
                )
            )
        return self.method_dict[loader_name]


class Opac_cia(Opac):
    """
    this class holds the opacity data and provides methods for evaluating the opacity at a given temperature and pressure.

    Everything's already been loaded into memory, so this is just a wrapper for the data.

    todo: add exotransmit CIA
    """

    method_dict = {
        "platon_cia": loader_platon_cia,
    }

    def __init__(self, filename, loader="chimera"):
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
        self.wl, self.T, self.cross_section = load_obj.load(filename, loader=loader)

        self.n_wav, self.n_t = len(self.wl), len(self.T)

        return
