from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from map2d import gridshow

cmap2d = CMap2D.CMap2D(".", "default")

# Dijkstra
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
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
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
pgrid4 = cmap2d.dijkstra(ij[0], inv_value=1000, connectedness=32)
pgrid8 = cmap2d.dijkstra(ij[0], inv_value=1000, connectedness=32)
pgrid16 = cmap2d.dijkstra(ij[0], inv_value=1000, connectedness=32)
pgrid32 = cmap2d.dijkstra(ij[0], inv_value=1000, connectedness=32)

tic = timer()
grid4path4,  jumps = CMap2D.path_from_dijkstra_field(pgrid4, [0, 0], connectedness= 4)
toc = timer()
grid4path8,  jumps = CMap2D.path_from_dijkstra_field(pgrid4, [0, 0], connectedness= 8)
grid4path16, jumps = CMap2D.path_from_dijkstra_field(pgrid4, [0, 0], connectedness=16)
grid4path32, jumps = CMap2D.path_from_dijkstra_field(pgrid4, [0, 0], connectedness=32)
print("Dijkstra descent 4-connected : {} ms".format((toc-tic)*0.001))
tic = timer()
grid8path4,  jumps = CMap2D.path_from_dijkstra_field(pgrid8, [0, 0], connectedness= 4)
toc = timer()
grid8path8,  jumps = CMap2D.path_from_dijkstra_field(pgrid8, [0, 0], connectedness= 8)
grid8path16, jumps = CMap2D.path_from_dijkstra_field(pgrid8, [0, 0], connectedness=16)
grid8path32, jumps = CMap2D.path_from_dijkstra_field(pgrid8, [0, 0], connectedness=32)
print("Dijkstra descent 8-connected : {} ms".format((toc-tic)*0.001))
tic = timer()
grid16path4,  jumps = CMap2D.path_from_dijkstra_field(pgrid16, [0, 0], connectedness= 4)
toc = timer()
grid16path8,  jumps = CMap2D.path_from_dijkstra_field(pgrid16, [0, 0], connectedness= 8)
grid16path16, jumps = CMap2D.path_from_dijkstra_field(pgrid16, [0, 0], connectedness=16)
grid16path32, jumps = CMap2D.path_from_dijkstra_field(pgrid16, [0, 0], connectedness=32)
print("Dijkstra descent 16-connected : {} ms".format((toc-tic)*0.001))
tic = timer()
grid32path4,  jumps = CMap2D.path_from_dijkstra_field(pgrid32, [0, 0], connectedness= 4)
toc = timer()
grid32path8,  jumps = CMap2D.path_from_dijkstra_field(pgrid32, [0, 0], connectedness= 8)
grid32path16, jumps = CMap2D.path_from_dijkstra_field(pgrid32, [0, 0], connectedness=16)
grid32path32, jumps = CMap2D.path_from_dijkstra_field(pgrid32, [0, 0], connectedness=32)
print("Dijkstra descent 32-connected : {} ms".format((toc-tic)*0.001))

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
plt.sca(ax0)
gridshow(grid4)
plt.plot(grid4path4[:,0],  grid4path4[:,1] , '-o', color=plt.cm.winter(  0.) , mec=plt.cm.winter(  0.), mfc=[0,0,0,0])
plt.plot(grid4path8[:,0],  grid4path8[:,1] , '-o', color=plt.cm.winter(0.33) , mec=plt.cm.winter(0.33), mfc=[0,0,0,0])
plt.plot(grid4path16[:,0], grid4path16[:,1], '-o', color=plt.cm.winter(0.66) , mec=plt.cm.winter(0.66), mfc=[0,0,0,0])
plt.plot(grid4path32[:,0], grid4path32[:,1], '-o', color=plt.cm.winter(  1.) , mec=plt.cm.winter(  1.), mfc=[0,0,0,0])
plt.title("4-connected")
plt.sca(ax1)
gridshow(grid8)
plt.plot(grid8path4[:,0],  grid8path4[:,1] , '-o', color=plt.cm.winter(  0.) , mec=plt.cm.winter(  0.), mfc=[0,0,0,0])
plt.plot(grid8path8[:,0],  grid8path8[:,1] , '-o', color=plt.cm.winter(0.33) , mec=plt.cm.winter(0.33), mfc=[0,0,0,0])
plt.plot(grid8path16[:,0], grid8path16[:,1], '-o', color=plt.cm.winter(0.66) , mec=plt.cm.winter(0.66), mfc=[0,0,0,0])
plt.plot(grid8path32[:,0], grid8path32[:,1], '-o', color=plt.cm.winter(  1.) , mec=plt.cm.winter(  1.), mfc=[0,0,0,0])
plt.title("8-connected")
plt.sca(ax2)
gridshow(grid16)
plt.plot(grid16path4[:,0],  grid16path4[:,1] , '-o', color=plt.cm.winter(  0.) , mec=plt.cm.winter(  0.), mfc=[0,0,0,0])
plt.plot(grid16path8[:,0],  grid16path8[:,1] , '-o', color=plt.cm.winter(0.33) , mec=plt.cm.winter(0.33), mfc=[0,0,0,0])
plt.plot(grid16path16[:,0], grid16path16[:,1], '-o', color=plt.cm.winter(0.66) , mec=plt.cm.winter(0.66), mfc=[0,0,0,0])
plt.plot(grid16path32[:,0], grid16path32[:,1], '-o', color=plt.cm.winter(  1.) , mec=plt.cm.winter(  1.), mfc=[0,0,0,0])
plt.title("16-connected")
plt.sca(ax3)
gridshow(grid32/cmap2d.resolution())
plt.plot(grid32path4[:,0],  grid32path4[:,1] , '-o', color=plt.cm.winter(  0.) , mec=plt.cm.winter(  0.), mfc=[0,0,0,0])
plt.plot(grid32path8[:,0],  grid32path8[:,1] , '-o', color=plt.cm.winter(0.33) , mec=plt.cm.winter(0.33), mfc=[0,0,0,0])
plt.plot(grid32path16[:,0], grid32path16[:,1], '-o', color=plt.cm.winter(0.66) , mec=plt.cm.winter(0.66), mfc=[0,0,0,0])
plt.plot(grid32path32[:,0], grid32path32[:,1], '-o', color=plt.cm.winter(  1.) , mec=plt.cm.winter(  1.), mfc=[0,0,0,0])
plt.title("32-connected")
ax3.legend([ '4-connected', '8-connected', '16-connected', '32-connected'])

# ax3.plot(path[:,0], path[:,1], '-,r')
# ax3.plot(path[:,0], path[:,1], '-,b')

# Contours as Vertices
tic = timer()
contours = cmap2d.as_closed_obst_vertices()
toc=timer()
print("Contours: {} ms".format((toc-tic)*0.001))
# print("plotting contours... ")
# plt.figure()
# gridshow(cmap2d.occupancy())
# contours_ij = [cmap2d.xy_to_ij(c) for c in contours]
# cmap2d.plot_contours(contours_ij, '-,k')
# print("done.")


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
