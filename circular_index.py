import numpy as np
from timeit import default_timer as timer


def make_circular_index(pixel_radius, sample_factor=10):
    """ Creates a circular index around the point at index 0, 0
    This method is sample-based and therefore not precise. 
    Solution is found by sampling at regular interval along the circle
    and storing the resulting points in 2d grid of resolution 1.
    Points are returned in order, forming a 2d grid circular path."""
    px_r = pixel_radius # radius of the circle in pixels

    # ensures sample_factor samples per angular pixel.
    kNSamples = max(12, 2 * np.pi * px_r * sample_factor) 

    theta = np.arange(0, kNSamples) * 2 * np.pi / kNSamples
    x = px_r * np.cos(theta)
    y = px_r * np.sin(theta)
    i = np.round(x).astype(int)
    j = np.round(y).astype(int)
    ij = np.vstack([i,j])
    unique_id = np.sort(np.unique(ij, axis=1, return_index=True)[1])
    i = ij[0, unique_id]
    j = ij[1, unique_id]
    return i, j

def index_to_grid(i, j, gridsize=None):
    """ Converts the circle from index format to 2d-grid format """
    size = gridsize
    if gridsize is None:
        size_i = max(i) - min(i) + 3
        size_j = max(j) - min(j) + 3
        size = max(size_i, size_j)
    mid = int((size - 1) / 2)
    grid = np.zeros((size, size))
    grid[(i+mid, j+mid)] = 1
    return grid

class CircularIndexCreator(object):
    """ Allows creating circular indices for given radii,
    and stores the results in a lookup table to avoid repeat computation """
    def __init__(self):
        self.px_r_archive = {}
        self.indices_archive = {}

    def make_circular_index(self, radius, resolution=1.):
        r = radius # [m]
        res = resolution # [m / px]
        px_r = radius / resolution # pixel-radius of circle [px]

        if px_r in self.px_r_archive:
            i, j = self.px_r_archive[px_r]
            return i, j

        i, j = make_circular_index(px_r)
        # store result in lookup table
        self.px_r_archive[px_r] = (i, j)
        return i, j

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    N_CIRCLES = 1000
    PLOT_RADII = False
    PLOT_EVOLUTION = True
    px_radii = np.linspace(0, 100, N_CIRCLES)
    cc = CircularIndexCreator()
    circle_lengths = []
    t_st = timer()
    for px_r in px_radii:
        i, j = cc.make_circular_index(px_r, 1)
        circle_lengths.append(len(i))
    t_end = timer()
    print("First-time computation of {} circles took {} seconds.".format(N_CIRCLES, t_end - t_st))
    t_st = timer()
    for px_r in px_radii:
        i, j = cc.make_circular_index(px_r, 1)
    t_end = timer()
    print("Second-time computation of {} circles took {} seconds.".format(N_CIRCLES, t_end - t_st))
    if PLOT_RADII:
        plt.figure()
        for px_r in px_radii:
            i, j = cc.make_circular_index(px_r, 1)
            plt.plot(i, j, '-x')
        plt.show()
        plt.figure()
        plt.plot(px_radii, circle_lengths, '-x')
        plt.show()
    if PLOT_EVOLUTION:
        print("Plotting evolution of circles with radius")
        N = 20
        px_radii = np.logspace(np.log(0.5),np.log10(N-1),200)
        for px_r in px_radii:
            i, j = cc.make_circular_index(px_r, 1)
        plt.ion()
        plt.figure()
        for px_r in px_radii:
            plt.gca().clear()
            i, j = cc.make_circular_index(px_r, 1)
            grid = index_to_grid(i, j, 2*N+1)
            plt.imshow(1-grid, cmap='gray')
            plt.gca().add_patch(plt.Circle((N,N), radius=px_r, color='white', fill=False))
            plt.pause(0.005)
        plt.ioff()
