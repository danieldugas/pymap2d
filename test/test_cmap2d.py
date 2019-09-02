from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from map2d import gridshow
from pyniel.testing_tools.testgen import load_test_case

import unittest

class TestStringMethods(unittest.TestCase):

    def test_office_full_coarse(self):
        cmap2d = CMap2D.CMap2D(".", "office_full", silent=True)
        coarse = cmap2d.as_coarse_map2d(n=3)
        test_data = load_test_case(folder="test_data", case_name="office_full_coarse")
        self.assertTrue(np.allclose(test_data['outputs'], coarse.occupancy()))

    def test_office_full_coarse_as_sdf(self):
        cmap2d = CMap2D.CMap2D(".", "office_full", silent=True)
        coarse = cmap2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()
        test_data = load_test_case(folder="test_data", case_name="office_full_coarse_as_sdf")
        self.assertTrue(np.allclose(test_data['outputs'], coarse.as_sdf()))

    def test_default_dijkstra_16(self):
        cmap2d = CMap2D.CMap2D(".", "tests", silent=True)
        test_data = load_test_case(folder="test_data", case_name="default_dijkstra_16")
        args = test_data['args']
        kwargs = test_data['kwargs']
        self.assertTrue(np.allclose(test_data['outputs'], cmap2d.dijkstra(*args, **kwargs)))

    def test_default_dijkstra_32(self):
        cmap2d = CMap2D.CMap2D(".", "tests", silent=True)
        test_data = load_test_case(folder="test_data", case_name="default_dijkstra_32")
        args = test_data['args']
        kwargs = test_data['kwargs']
        self.assertTrue(np.allclose(test_data['outputs'], cmap2d.dijkstra(*args, **kwargs)))

    def test_coord_transforms(self):
        cmap2d = CMap2D.CMap2D(".", "default", silent=True)
        ij = cmap2d.xy_to_ij(np.array([[0, 0], [1,1]], dtype=np.float32))
        cmap2d.ij_to_xy(ij.astype(np.float32))

    def test_dijkstra(self):
        cmap2d = CMap2D.CMap2D(".", "default", silent=True)
        # setting an invalid / masked point as goal should not segfault
        cmap2d.dijkstra(np.array([31,36]))

if __name__ == '__main__':
    unittest.main()
