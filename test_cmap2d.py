import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def gridshow(*args, **kwargs):
    """ utility function for showing 2d grids in matplotlib """
    from matplotlib import pyplot as plt
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'
    if not 'cmap' in kwargs:
        kwargs['cmap'] = plt.cm.Greys
    return plt.imshow(*(arg.T if i == 0 else arg for i, arg in enumerate(args)), **kwargs)

cmap2d = CMap2D.CMap2D(".", "office_full")
coarse = cmap2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()
ij = cmap2d.xy_to_ij(np.array([[0, 0], [1,1]], dtype=np.float32))
print(ij)
xy = cmap2d.ij_to_xy(ij.astype(np.float32))
print(xy)

# occupancy
plt.figure() 
grid = cmap2d.occupancy()
gridshow(grid)

# as Vertices
plt.figure()
contours = cmap2d.as_closed_obst_vertices()
gridshow(cmap2d.occupancy())
cmap2d.plot_contours(contours)

# SDF
plt.figure()
tic = timer()
grid = cmap2d.as_sdf()
a = grid
toc = timer()
print("SDF: {} ms".format((toc-tic)*0.001))
gridshow(grid)

# TSDF
plt.figure()
tic = timer()
grid = cmap2d.as_tsdf(0.5)
toc = timer()
print("TSDF : {} ms".format((toc-tic)*0.001))
gridshow(grid)

# Dijkstra
plt.figure()
tic = timer()
ij = coarse.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
grid = coarse.dijkstra(ij[0], inv_value=-1)
toc = timer()
print("Dijsktra : {} ms".format((toc-tic)*0.001))
gridshow(grid)
plt.show()
