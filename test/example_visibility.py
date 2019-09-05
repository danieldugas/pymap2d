from __future__ import print_function
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
Greys_c10 = ListedColormap(plt.cm.Greys(np.mod(np.linspace(0., 10., 256), 1.)))
from timeit import default_timer as timer

from map2d import gridshow

cmap2d = CMap2D.CMap2D(".", "tests")
# ij = cmap2d.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
ij = np.array([[63,63]])
cmap2d.set_resolution(1.)

plt.figure()
tic = timer()
vis = cmap2d.visibility_map([0, 0])
toc = timer()
print("Vis : {} ms".format((toc-tic)*1000))
gridshow(vis + 100*cmap2d.occupancy())

plt.figure()
tic = timer()
vis = cmap2d.visibility_map([40, 30])
toc = timer()
print("Vis : {} ms".format((toc-tic)*1000))
gridshow(vis + 100*cmap2d.occupancy())
plt.show()
