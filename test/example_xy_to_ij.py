from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt

from map2d import gridshow

cmap2d = CMap2D.CMap2D(".", "office_full")
coarse = cmap2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()
# extent
imax, jmax = cmap2d.occupancy().shape
ijcorners = np.array([[0,0], [imax-1, jmax-1]])
xycorners = cmap2d.ij_to_xy(ijcorners)
extent = [xycorners[0, 0], xycorners[1, 0], xycorners[0, 1], xycorners[1, 1]]

# occupancy
plt.figure("ij")
grid = cmap2d.occupancy()
plt.xlabel("i")
plt.ylabel("j")
plt.title("Pixel Coordinates")
gridshow(grid)


# visibility
plt.figure("xy")
grid = cmap2d.occupancy()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Spatial Coordinates")
gridshow(grid, extent=extent)

plt.show()
