from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from map2d import gridshow

import unittest

class TestStringMethods(unittest.TestCase):

    def test_coord_transforms(self):
        cmap2d = CMap2D.CMap2D(".", "default")
        ij = cmap2d.xy_to_ij(np.array([[0, 0], [1,1]], dtype=np.float32))
        cmap2d.ij_to_xy(ij.astype(np.float32))

    def test_dijkstra(self):
        cmap2d = CMap2D.CMap2D(".", "default")
        # setting an invalid / masked point as goal should not segfault
        cmap2d.dijkstra(np.array([31,36]))

if __name__ == '__main__':
    unittest.main()
