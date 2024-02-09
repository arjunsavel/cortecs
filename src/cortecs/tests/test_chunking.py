"""
just do a few tests for chunking...
"""
import unittest
import os
import numpy as np
from cortecs.opac.chunking import chunk_wavelengths
from cortecs.opac.opac import Opac


class TestIntegration(unittest.TestCase):
    opacity_file = os.path.abspath(".") + "/src/cortecs/tests/temperatures.npy"

    def test_chunking_two_files(self):
        """
        Test the chunking function.
        """
        chunk_wavelengths(self.opacity_file, nchunks=2)  # evenly split into two chunks

        # now test that the two files were created

        self.assertTrue(
            os.path.exists(
                os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl0.dat"
            )
            and os.path.exists(
                os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl1.dat"
            )
        )

    def test_wls_of_each_created_file(self):
        """
        Test that the number of wavelengths in each file is EXACTLY the same.
        """
        chunk_wavelengths(self.opacity_file, nchunks=2)

        # now get the wavelengths of each file
        opac_obj_ref = Opac(self.opacity_file, loader="exotransmit")
        opac_obj0 = Opac(
            os.path.abspath(".") + "/src/cortecs/tests/opacCH4_narrow_wl0.dat",
            loader="exotransmit",
        )
        opac_obj1 = Opac(
            os.path.abspath(".") + "/src/cortecs/tests/opacCH4_narrow_wl1.dat",
            loader="exotransmit",
        )

        np.testing.assert_array_equal(
            opac_obj_ref.wl, np.concatenate((opac_obj0.wl, opac_obj1.wl))
        )

        # self.assertEqual(opac_obj_ref.wl), len(opac_obj0.wl) + len(opac_obj1.wl))
