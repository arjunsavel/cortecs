"""
this file holds the class describing opacity data. hmm...maybe I want the loader in a different file?

author: @arjunsavel
"""
import numpy as np

from cortecs.opac.io import *


class Opac(object):
    """
    this class holds the opacity data and provides methods for evaluating the opacity at a given temperature and pressure.

    Everything's already been loaded into memory, so this is just a wrapper for the data.

    todo: use the different loaders
    """

    method_dict = {
        "chimera": loader_chimera,
        "helios": loader_helios,
        "platon": loader_platon,
        "exotransmit": loader_exotransmit,
    }

    wl = None
    T = None
    P = None

    def __init__(
        self, filename, loader="chimera", load_kwargs=None, loader_kwargs=None
    ):
        """
        wraps around the loaders.

        Parameters
        ----------
        filename : str
            name of file to load
        loader : str
            name of loader to use. default is chimera.
        load_kwargs : dict
            keyword arguments to pass to the loader when loading.
        loader_kwargs : dict
            keyword arguments to pass to the loader when  initializing.

        Returns
        -------
        nothing
        """
        if load_kwargs is None:
            load_kwargs = {}
        if loader_kwargs is None:
            loader_kwargs = {}

        self.filename = filename
        self.load_obj = self._get_loader(loader, **loader_kwargs)
        self.wl, self.T, self.P, self.cross_section = self.load_obj.load(
            filename, **load_kwargs
        )

        self.n_wav, self.n_t, self.n_p = len(self.wl), len(self.T), len(self.P)

        return

    def _get_loader(self, loader_name, **kwargs):
        """
        gets the loader object

        Parameters
        ----------
        loader_name : str
            name of loader
        kwargs : dict
            keyword arguments to pass to the loader.

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
        return self.method_dict[loader_name](**kwargs)

    # todo: implement the copy and deepcopy methods.
    def copy(self):
        # need a new opac object with all the same attributes. so let's make sure we don't load on input.
        raise NotImplementedError


class Opac_cia(Opac):
    """
    this class holds the opacity data and provides methods for evaluating the opacity at a given temperature and pressure.

    Everything's already been loaded into memory, so this is just a wrapper for the data.

    """

    method_dict = {
        "platon_cia": loader_platon_cia,
        "exotransmit_cia": loader_exotransmit_cia,
    }

    def __init__(self, filename, loader="platon_cia", view="default"):
        """
        wraps around the loaders.
        NOTE: there's a mixed type for self.cross_section. can be a dataframe if there are a number of species.

        Parameters
        ----------
        filename : str
            name of file to load
        loader : str
            name of loader to use. default is chimera.
        view : str
            if 'full_frame' is selected, the cross_section object also includes temperature and wavelength columns.

        Returns
        -------
        nothing
        """
        self.filename = filename
        self.load_obj = self._get_loader(loader)
        self.wl, self.T, self.cross_section = self.load_obj.load(self.filename)

        # at least for the exotransmit case, we have...wl x temp.
        if view == "full_frame":
            self.cross_section["temp"] = self.T
            self.cross_section["wav"] = self.wl

        self.n_wav, self.n_t = len(self.wl), len(self.T)

        return

    def join_cross_section(self, opac):
        """
        joins another opacity's cross-section data to this one.
        :param opac:
        :return:
        """

        def replace_zeros(species):
            mask = self.cross_section[species] == 0.0
            self.cross_section.loc[mask, species] = opac.cross_section.loc[
                mask, species
            ]

        # throw errors if the grids don't match.
        if np.all(opac.T != self.T):
            raise ValueError("temperatures do not match!")
        if np.all(opac.wl != self.wl):
            raise ValueError("wavelengths do not match!")

        # add the entirely new columns
        other_columns = opac.cross_section.columns
        for column in other_columns:
            if column not in self.cross_section.columns:
                self.cross_section[column] = opac.cross_section[column]

        # now fill in any temperatures where the cross-section is zero here.
        for species in self.cross_section.columns:
            if species in other_columns:
                replace_zeros(species)

        return
