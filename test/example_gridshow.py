import numpy as np
import matplotlib.pyplot as plt
from CMap2D import gridshow, CMap2D

resolution = 1.
origin = (-5., -5)
occupancy = np.zeros((12, 12))
occupancy[2:4, 8:10] = 1.
occupancy[8:10, 8:9] = 1.
cmap2d = CMap2D()
cmap2d.from_array(occupancy, origin, resolution)
ii, jj = cmap2d.as_meshgrid_ij()
xx, yy = cmap2d.as_meshgrid_xy()
rr = np.sqrt((xx-1) * (xx-1) + yy * yy)
smile = np.logical_and.reduce((
    rr > 3,
    rr < 5,
    yy < 0,
))
occupancy[np.where(smile)] = 1.
cmap2d.from_array(occupancy, origin, resolution)


fig, (ax1, ax2) = plt.subplots(1, 2)
plt.sca(ax1)
gridshow(cmap2d.occupancy(), extent=cmap2d.get_extent_xy())
plt.title("gridshow")
plt.xlabel("x")
plt.ylabel("y")
plt.sca(ax2)
plt.title("plt.imshow")
e = cmap2d.get_extent_xy()
imshow_extent = [e[2], e[3], e[1], e[0]]
plt.imshow(cmap2d.occupancy(), extent=imshow_extent)
plt.xlabel("y")
plt.ylabel("x (descending)")

plt.show()
