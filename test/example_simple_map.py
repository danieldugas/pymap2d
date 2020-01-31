from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from map2d import gridshow

cmap2d = CMap2D.CMap2D().as_coarse_map2d(3)
cmap2d._occupancy[2:4, 4:5] = 1.
coarse = cmap2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d()

# occupancy
plt.figure("occupancy") 
grid = cmap2d.occupancy()
gridshow(grid)

# visibility
plt.figure("visibility_map")
tic = timer()
vis = cmap2d.visibility_map([0,0])
toc = timer()
print("Visibility: {} ms, {} Hz".format((toc-tic)*1000, 1./(toc-tic)))
gridshow(vis)

# SDF
plt.figure("sdf")
tic = timer()
grid = cmap2d.as_sdf()
a = grid
toc = timer()
print("SDF: {} ms".format((toc-tic)*1000))
gridshow(grid)

# TSDF
plt.figure("tsdf")
tic = timer()
grid = cmap2d.as_tsdf(0.5)
toc = timer()
print("TSDF : {} ms".format((toc-tic)*1000))
gridshow(grid)

# Dijkstra
plt.figure("dijkstra")
tic = timer()
ij = coarse.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
grid8 = coarse.dijkstra(ij[0], inv_value=-1)
toc = timer()
print("Dijsktra : {} ms".format((toc-tic)*1000))

tic = timer()
ij = coarse.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
grid16 = coarse.dijkstra(ij[0], inv_value=-1, connectedness=16)
toc = timer()
print("Dijsktra 16-connected : {} ms".format((toc-tic)*1000))

tic = timer()
ij = coarse.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
grid32 = coarse.dijkstra(ij[0], inv_value=-1, connectedness=32)
toc = timer()
print("Dijsktra 32-connected : {} ms".format((toc-tic)*1000))
gridshow(grid32)
# plt.contour(grid8.T, colors='red')
# plt.contour(grid32.T, colors='blue')

grid = coarse.dijkstra(ij[0], inv_value=1000, connectedness=32)
tic = timer()
path, jumps = CMap2D.path_from_dijkstra_field(grid, [0, 0])
toc = timer()
print("Dijkstra descent : {} ms".format((toc-tic)*1000))
plt.plot(path[:,0], path[:,1], '-,r')

tic = timer()
path, jumps = CMap2D.path_from_dijkstra_field(grid, [0, 0], connectedness=32)
toc = timer()
print("Dijkstra descent 32-connected : {} ms".format((toc-tic)*1000))
plt.plot(path[:,0], path[:,1], '-,b')

# Contours as Vertices
tic = timer()
contours = cmap2d.as_closed_obst_vertices()
toc=timer()
print("Contours: {} ms".format((toc-tic)*1000))
print("plotting contours... ")
plt.figure("contours")
gridshow(cmap2d.occupancy())
contours_ij = [cmap2d.xy_to_ij(c) for c in contours]
cmap2d.plot_contours(contours_ij)
print("done.")

plt.show()

