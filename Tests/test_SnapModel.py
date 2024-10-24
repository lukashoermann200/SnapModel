"""This module contains unit tests for all functions/classes contained
in the module GPRLib.py.

Author:
    - Johannes Cartus, TU Graz, 03.07.2019
"""

import numpy as np
import unittest
import pytest

import aimstools.Utilities as ut
import numpy as np

from aimstools.GeometryFile import GeometryFile

from SnapModel.SnapModel import (
    EnergyModelML,
    EnergyModelLJ,
    SnapModel,
    getInitialValuesForPhiAndTheta,
)


class TestSnapModel(unittest.TestCase):

    def test_ThermalActivation(self):
        horizontal_offset = -0.51
        angle = 0.0

        geom = GeometryFile("test_data/substrate.in")
        tip = GeometryFile("test_data/molecule.in")

        # CC path
        pos_C1 = np.array([3.449, 8.636, 11.142])
        pos_C2 = np.array([4.530, 7.722, 11.142])
        osc_center = (pos_C1 + pos_C2) / 2

        vec_CC = pos_C1 - pos_C2
        osc_dir_0 = np.array([vec_CC[1], -vec_CC[0], 0])
        osc_dir_0 /= np.linalg.norm(osc_dir_0)

        R = ut.getCartesianRotationMatrix(
            angle * np.pi / 180, get_3x3_matrix=True
        )
        osc_dir = np.dot(osc_dir_0, R)

        f0 = 37000
        asc_amp = 0.5
        tors = 22e-21
        CO_length = 4
        k = 1343
        NSteps = 64
        apex_height = 7.1

        energy_model = EnergyModelLJ(geom)

        snap_model_TA = SnapModel(
            geom,
            tip,
            osc_center + horizontal_offset * osc_dir_0,
            osc_dir,
            asc_amp,
            CO_length,
            tors,
            k,
            energy_model,
            init_steps=2,
            optimizer="Nelder-Mead",
            themal_activation=True,
            dims=2,
        )

        th, phi = getInitialValuesForPhiAndTheta(osc_dir_0, osc_dir)
        df_TA, Ediss_TA = snap_model_TA.runOscillation(
            f0, NSteps, apex_height, th, phi
        )

        snap_model = SnapModel(
            geom,
            tip,
            osc_center + horizontal_offset * osc_dir_0,
            osc_dir,
            asc_amp,
            CO_length,
            tors,
            k,
            energy_model,
            init_steps=2,
            optimizer="Nelder-Mead",
            themal_activation=False,
            dims=2,
        )

        th, phi = getInitialValuesForPhiAndTheta(osc_dir_0, osc_dir)
        df, Ediss = snap_model.runOscillation(f0, NSteps, apex_height, th, phi)

        self.assertEqual(Ediss_TA, Ediss)


if __name__ == "__main__":
    unittest.main()
