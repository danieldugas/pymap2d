import CMap2D
import numpy as np
import matplotlib.pyplot as plt

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
gridshow(cmap2d.occupancy())
plt.show()
gridshow(cmap2d.as_sdf())
plt.show()
ij = coarse.xy_to_ij(np.array([[0, 0]], dtype=np.float32))
gridshow(coarse.dijkstra(ij[0], inv_value=-1))
plt.show()
