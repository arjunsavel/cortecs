"""
tests that the opacity object behaves as expected.
"""
import unittest

from cortecs.opac.opac import *
import copy
import os

import numpy as np
from cortecs.opac.interpolate_cia import *
import io
import sys


class TestOpac(unittest.TestCase):
    """
    tests the Opac object.
    """

    cia_filename = os.path.abspath(".") + "/src/cortecs/tests/test_opac_cia.dat"

    def test_wrong_method(self):
        """
        If I pass a method that isn't sypported, an error should be raised.
        :return:
        """
        with self.assertRaises(ValueError):
            _ = Opac_cia(self.cia_filename, loader="wrongmethod", view="full_frame")

    def test_cia_opac_instantiated(self):
        """
        does an error get thrown when I'm just trying to make a little opacCIA object?
        :return:
        """
        _ = Opac_cia(self.cia_filename, loader="exotransmit_cia", view="full_frame")
        self.assertTrue(True)

    def test_cia_opac_join(self):
        """
        does an error get thrown when I'm just trying to join two things?
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test2.join_cross_section(opac_test)
        self.assertTrue(True)

    def test_cia_opac_join_no_add(self):
        """
        shouldn't add any new columns if the second one has fewer columns.
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test2.join_cross_section(opac_test)
        self.assertTrue(
            len(opac_test2.cross_section.columns)
            == len(opac_test.cross_section.columns)
        )

    def test_cia_opac_join_yes_add(self):
        """
        shouldn't ONLY add the new columns if the second one has more columns.
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        initial_copy = copy.deepcopy(opac_test2)
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test.cross_section["other_col"] = opac_test.cross_section["temp"] * 13
        opac_test2.join_cross_section(opac_test)

        self.assertTrue(
            len(initial_copy.cross_section.columns) + 1
            == len(opac_test2.cross_section.columns)
        )

    def test_cia_opac_diff_temp(self):
        """
        needs same temp grid for joining.
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test.T = [23]
        self.assertRaises(ValueError, opac_test2.join_cross_section, opac_test)

    def test_cia_opac_diff_wl(self):
        """
        needs same wl grid
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test.wl = [23]
        self.assertRaises(ValueError, opac_test2.join_cross_section, opac_test)

    def test_cia_opac_final_grid(self):
        """
        need final temperature grid to be chill
        :return:
        """
        opac_test2 = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        initial_copy = copy.copy(opac_test2)
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        opac_test.cross_section["other_col"] = opac_test.cross_section["temp"] * 13
        opac_test2.join_cross_section(opac_test)
        self.assertTrue(
            len(initial_copy.cross_section) == len(opac_test2.cross_section)
        )


class TestIO(unittest.TestCase):
    cia_filename = os.path.abspath(".") + "/src/cortecs/tests/test_opac_cia.dat"

    def test_write_cia_same_cross_section(self):
        """
        when i write a cia and read it back in, is it the same?
        :return:
        """
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        writer = writer_exotransmit_cia()

        writer.write(opac_test, "test_written_cia.dat")

        opac_reread = Opac_cia(
            "test_written_cia.dat", loader="exotransmit_cia", view="full_frame"
        )
        self.assertTrue(opac_test.cross_section.equals(opac_reread.cross_section))

    def test_write_cia_same_temp(self):
        """
        when i write a cia and read it back in, is it the same?
        :return:
        """
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        writer = writer_exotransmit_cia()

        writer.write(opac_test, "test_written_cia.dat")

        opac_reread = Opac_cia(
            "test_written_cia.dat", loader="exotransmit_cia", view="full_frame"
        )
        self.assertTrue(
            opac_test.cross_section.temp.equals(opac_reread.cross_section.temp)
            and np.all(opac_test.T == opac_reread.T)
        )

    def test_write_cia_same_wl(self):
        """
        when i write a cia and read it back in, is it the same?
        :return:
        """
        opac_test = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        writer = writer_exotransmit_cia()

        writer.write(opac_test, "test_written_cia.dat")

        opac_reread = Opac_cia(
            "test_written_cia.dat", loader="exotransmit_cia", view="full_frame"
        )
        self.assertTrue(
            opac_test.cross_section.wav.equals(opac_reread.cross_section.wav)
            and np.all(opac_test.wl == opac_reread.wl)
        )


class TestInterpolateCIA(unittest.TestCase):
    T_filename = os.path.abspath(".") + "/src/cortecs/tests/temperatures.npy"
    P_filename = os.path.abspath(".") + "/src/cortecs/tests/pressures.npy"
    wl_filename = os.path.abspath(".") + "/src/cortecs/tests/wavelengths.npy"
    cross_sec_filename = (
        os.path.abspath(".") + "/src/cortecs/tests/absorb_coeffs_C2H4.npy"
    )
    cia_filename = os.path.abspath(".") + "/src/cortecs/tests/test_opac_cia.dat"

    def test_interpolate_cia_within_bounds(self):
        """
        I'm linearly interpolating. If I'm within the bounds of the data, I should get the same answer.
        :return:
        """
        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        interpolate_cia(
            self.cia_filename,
            self.cross_sec_filename,
            loader="platon",
            outfile="test_interpolate.dat",
            load_kwargs=load_kwargs,
        )
        interpolated_cia = Opac_cia(
            "test_interpolate.dat", loader="exotransmit_cia", view="full_frame"
        )
        original_cia = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )

        # find nearest neighbor to one value
        test_val = interpolated_cia.cross_section["H2H2s"].values[-20]
        test_wl = 2.436485e-06
        test_temp = 2900
        test_val = 3.132829e-54

        # find nearest wavelength above
        closest_above_wl = original_cia.cross_section["wav"][
            original_cia.cross_section["wav"] > test_wl
        ].min()
        # find nearest neighbor below
        closest_below_wl = original_cia.cross_section["wav"][
            original_cia.cross_section["wav"] < test_wl
        ].max()
        # import pdb
        # pdb.set_trace()
        closest_above_val = original_cia.cross_section["H2H2s"][
            (original_cia.cross_section["wav"] == closest_above_wl)
            & (original_cia.cross_section["temp"] == test_temp)
        ].values[0]
        closest_below_val = original_cia.cross_section["H2H2s"][
            (original_cia.cross_section["wav"] == closest_below_wl)
            & (original_cia.cross_section["temp"] == test_temp)
        ].values[0]
        # check that the interpolated value is between the two.
        self.assertTrue(
            closest_below_val <= test_val <= closest_above_val
            or closest_below_val >= test_val >= closest_above_val
        )

    def test_cia_temperature_check(self):
        """
        the temperature check should go correctly!
        :return:
        """
        original_cia = Opac_cia(
            self.cia_filename, loader="exotransmit_cia", view="full_frame"
        )
        capturedOutput = io.StringIO()  # Create StringIO.
        sys.stdout = capturedOutput  # Redirect stdout.
        check_temp_grid(original_cia.cross_section, [-20], "namename")  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        expected_string = "Temperature -20 not in CIA file namename! Cannot interpolate in temperature yet. Will set these values to 0.\n"
        output = capturedOutput.getvalue()

        print(output)
        print(expected_string)
        self.assertTrue(output == expected_string)

    def test_cia_in_out_temp_check(self):
        """
        when interpolating, i should have the same temperature grid as the reference opacity file.
        :return:
        """
        load_kwargs = {
            "T_filename": self.T_filename,
            "P_filename": self.P_filename,
            "wl_filename": self.wl_filename,
        }
        interpolate_cia(
            self.cia_filename,
            self.cross_sec_filename,
            loader="platon",
            outfile="test_interpolate.dat",
            load_kwargs=load_kwargs,
        )
        interpolated_cia = Opac_cia(
            "test_interpolate.dat", loader="exotransmit_cia", view="full_frame"
        )
        reference_opac = Opac(
            self.cross_sec_filename, loader="platon", load_kwargs=load_kwargs
        )
        self.assertTrue(np.all(np.unique(interpolated_cia.T) == reference_opac.T))
