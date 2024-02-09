"""
just do a few tests for chunking...
"""
import unittest
import os
import numpy as np
from cortecs.opac.chunking import chunk_wavelengths
from cortecs.opac.opac import Opac


class TestIntegration(unittest.TestCase):
    opacity_file = os.path.abspath(".") + "/src/cortecs/tests/opacCH4_narrow_wl.dat"
    first_file = os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl0.dat"
    second_file = (
        os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl1.dat"
    )

    def test_chunking_two_files(self):
        """
        Test the chunking function.
        """
        chunk_wavelengths(
            self.opacity_file, wav_per_chunk=2
        )  # evenly split into two chunks

        # now test that the two files were created

        self.assertTrue(
            os.path.exists(self.first_file) and os.path.exists(self.second_file)
        )

    def test_wls_of_each_created_file(self):
        """
        Test that the number of wavelengths in each file is EXACTLY the same.
        """
        # clean up the files already made!!
        os.remove(self.first_file)
        os.remove(self.second_file)

        chunk_wavelengths(self.opacity_file, wav_per_chunk=2)

        # now get the wavelengths of each file
        opac_obj_ref = Opac(self.opacity_file, loader="exotransmit")
        opac_obj0 = Opac(
            self.first_file,
            loader="exotransmit",
        )
        opac_obj1 = Opac(
            self.second_file,
            loader="exotransmit",
        )
        np.testing.assert_array_equal(
            opac_obj_ref.wl, np.concatenate((opac_obj0.wl, opac_obj1.wl))
        )

    def test_vals_of_each_created_file(self):
        """
        Test that the opacity values together are concatenated!
        """
        # clean up the files already made!!
        os.remove(self.first_file)
        os.remove(self.second_file)

        chunk_wavelengths(self.opacity_file, wav_per_chunk=2)

        # now get the wavelengths of each file
        opac_obj_ref = Opac(self.opacity_file, loader="exotransmit")
        opac_obj0 = Opac(
            self.first_file,
            loader="exotransmit",
        )
        opac_obj1 = Opac(
            self.second_file,
            loader="exotransmit",
        )
        # pdb.set_trace()
        np.testing.assert_array_equal(
            opac_obj_ref.cross_section,
            np.concatenate((opac_obj0.cross_section, opac_obj1.cross_section)),
        )

        # self.assertEqual(opac_obj_ref.wl), len(opac_obj0.wl) + len(opac_obj1.wl))
