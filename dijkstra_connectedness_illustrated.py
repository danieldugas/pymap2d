from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from map2d import gridshow

cmap2d = CMap2D.CMap2D(".", "default")

# Dijkstra
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
ij = cmap2d.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
# ij = np.array([[0,0]])

tic = timer()
grid4 = cmap2d.dijkstra(ij[0], inv_value=-1, connectedness=4)
toc = timer()
print("Dijsktra 4-connected : {} ms".format((toc-tic)*0.001))

tic = timer()
grid8 = cmap2d.dijkstra(ij[0], inv_value=-1, connectedness=8)
toc = timer()
print("Dijsktra 8-connected : {} ms".format((toc-tic)*0.001))

tic = timer()
grid16 = cmap2d.dijkstra(ij[0], inv_value=-1, connectedness=16)
toc = timer()
print("Dijsktra 16-connected : {} ms".format((toc-tic)*0.001))

tic = timer()
grid32 = cmap2d.dijkstra(ij[0], inv_value=-1, connectedness=32)
toc = timer()
print("Dijsktra 32-connected : {} ms".format((toc-tic)*0.001))

# Show
plt.sca(ax0)
gridshow(np.mod(np.maximum(grid4, 0),2))
plt.title("4-connected")
plt.sca(ax1)
gridshow(np.mod(np.maximum(grid8, 0),2))
plt.title("8-connected")
plt.sca(ax2)
gridshow(np.mod(np.maximum(grid16, 0),2))
plt.title("16-connected")
plt.sca(ax3)
gridshow(np.mod(np.maximum(grid32, 0),2))
plt.title("32-connected")

# Descent
grid = cmap2d.dijkstra(ij[0], inv_value=1000, connectedness=32)
tic = timer()
path, jumps = CMap2D.path_from_dijkstra_field(grid, [0, 0])
toc = timer()
print("Dijkstra descent : {} ms".format((toc-tic)*0.001))

tic = timer()
path, jumps = CMap2D.path_from_dijkstra_field(grid, [0, 0], connectedness=32)
toc = timer()
print("Dijkstra descent 32-connected : {} ms".format((toc-tic)*0.001))

# ax3.plot(path[:,0], path[:,1], '-,r')
# ax3.plot(path[:,0], path[:,1], '-,b')

# Contours as Vertices
tic = timer()
contours = cmap2d.as_closed_obst_vertices()
toc=timer()
print("Contours: {} ms".format((toc-tic)*0.001))
print("plotting contours... ")
plt.figure()
gridshow(cmap2d.occupancy())
contours_ij = [cmap2d.xy_to_ij(c) for c in contours]
cmap2d.plot_contours(contours_ij, '-,k')
print("done.")


def plotshow():
    plt.show()
    interval = 0.01
    try:
        while True:
            backend = plt.rcParams['backend']
            if backend in matplotlib.rcsetup.interactive_bk:
                figManager = matplotlib._pylab_helpers.Gcf.get_active()
                if figManager is not None:
                    canvas = figManager.canvas
                    if canvas.figure.stale:
                        canvas.draw()
                    canvas.start_event_loop(interval)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting wait loop.")
plotshow()
