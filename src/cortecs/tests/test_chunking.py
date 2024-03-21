"""
just do a few tests for chunking...
"""

import unittest
import os
import numpy as np
from cortecs.opac.chunking import chunk_wavelengths, add_overlap
from cortecs.opac.opac import Opac


class TestIntegration(unittest.TestCase):
    load_kwargs = {"fullfile": False}

    opacity_file = os.path.abspath(".") + "/src/cortecs/tests/opacCH4_narrow_wl.dat"
    first_file = os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl0.dat"
    second_file = (
        os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl1.dat"
    )
    file_base = os.path.abspath(".") + "/src/cortecs/tests/" + "opacCH4_narrow_wl"

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
        opac_obj_ref = Opac(
            self.opacity_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj0 = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1 = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
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
        opac_obj_ref = Opac(
            self.opacity_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj0 = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1 = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        # pdb.set_trace()
        np.testing.assert_array_equal(
            opac_obj_ref.cross_section,
            np.concatenate((opac_obj0.cross_section, opac_obj1.cross_section)),
        )

        # self.assertEqual(opac_obj_ref.wl), len(opac_obj0.wl) + len(opac_obj1.wl))

    def test_add_overlap_wl_increase_or_same(self):
        """
        Test that the overlap is added to the end of the first file.
        """
        # clean up the files already made!!
        # os.remove(self.first_file)
        # os.remove(self.second_file)

        chunk_wavelengths(self.opacity_file, wav_per_chunk=2)
        opac_obj0_orig = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1_orig = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )

        add_overlap(self.file_base, v_max=0.0, load_kwargs=self.load_kwargs)

        # now get the wavelengths of each file
        opac_obj_ref = Opac(
            self.opacity_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj0 = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1 = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        # pdb.set_trace()
        self.assertTrue(
            len(opac_obj0.wl) >= len(opac_obj0_orig.wl)
            and len(opac_obj1.wl) >= len(opac_obj1_orig.wl)
        )

    def add_overlap_with_single_overlap_point(self):
        """
        just overlap a single point and make sure that works.
        :return:
        """
        # os.remove(self.first_file)
        # os.remove(self.second_file)

        chunk_wavelengths(self.opacity_file, wav_per_chunk=2)
        opac_obj0_orig = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1_orig = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )

        # calculate the vmax so that one point is changed
        # velocity is C * delta lambda over lambda
        c = 3e8
        max_curr_lam = np.max(opac_obj0_orig.wl)
        v_max = c * (opac_obj1_orig.wl[0] - max_curr_lam) / max_curr_lam
        add_overlap(self.file_base, v_max=v_max, load_kwargs=self.load_kwargs)

        # now get the wavelengths of each file
        opac_obj_ref = Opac(
            self.opacity_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj0 = Opac(
            self.first_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        opac_obj1 = Opac(
            self.second_file, loader="exotransmit", load_kwargs=self.load_kwargs
        )
        self.assertTrue(
            len(opac_obj1.wl.min) == len(opac_obj0.wl.max())
            and len(opac_obj1.wl) == 1 + len(opac_obj1_orig.wl)
        )
