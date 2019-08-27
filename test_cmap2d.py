from __future__ import print_function
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
xy = cmap2d.ij_to_xy(ij.astype(np.float32))

# occupancy
plt.figure() 
grid = cmap2d.occupancy()
gridshow(grid)

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
grid = coarse.fastdijkstra(ij[0], inv_value=-1)
toc = timer()
print("Dijsktra : {} ms".format((toc-tic)*0.001))
gridshow(grid)

grid = coarse.fastdijkstra(ij[0], inv_value=1000)
tic = timer()
path, jumps = CMap2D.path_from_dijkstra_field(grid, [50, 80])
toc = timer()
print("Dijkstra descent : {} ms".format((toc-tic)*0.001))
plt.plot(path[:,0], path[:,1], '-,r')

# Contours as Vertices
tic = timer()
contours = cmap2d.as_closed_obst_vertices()
toc=timer()
print("Contours: {} ms".format((toc-tic)*0.001))
print("plotting contours... ")
plt.figure()
gridshow(cmap2d.occupancy())
contours_ij = [cmap2d.xy_to_ij(c) for c in contours]
cmap2d.plot_contours(contours_ij)
print("done.")

plt.show()
