# distutils: language=c++

from libcpp cimport bool
from libcpp.queue cimport priority_queue as cpp_priority_queue
from libcpp.pair cimport pair as cpp_pair
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport acos as cacos
from libc.math cimport sqrt as csqrt
from libc.math cimport floor as cfloor

import os
from yaml import load
from matplotlib.pyplot import imread


cdef class CMap2D:
    cdef public np.float32_t[:,::1] _occupancy # [:, ::1] means 2d c-contiguous
    cdef int occupancy_shape0
    cdef int occupancy_shape1
    cdef float resolution_
    cdef float _thresh_occupied
    cdef float thresh_free
    cdef float HUGE_
    cdef public np.float32_t[:] origin
    def __init__(self, folder=None, name=None, silent=False):
        self._occupancy = np.ones((100, 100), dtype=np.float32) * 0.5
        self.occupancy_shape0 = 100
        self.occupancy_shape1 = 100
        self.resolution_ = 0.01
        self.origin = np.array([0., 0.], dtype=np.float32)
        self._thresh_occupied = 0.9
        self.thresh_free = 0.1
        self.HUGE_ = 1e10
        if folder is None or name is None:
            return
        # Load map from file
        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")
        if not silent:
            print("Loading map definition from {}".format(yaml_file))
        with open(yaml_file) as stream:
            mapparams = load(stream)
        map_file = os.path.join(folder, mapparams["image"])
        if not silent:
            print("Map definition found. Loading map from {}".format(map_file))
        mapimage = imread(map_file)
        temp = (1. - mapimage.T[:, ::-1] / 254.).astype(np.float32)
              # (0 to 1) 1 means 100% certain occupied
        mapimage = np.ascontiguousarray(temp)
        self._occupancy = mapimage
        self.occupancy_shape0 = mapimage.shape[0]
        self.occupancy_shape1 = mapimage.shape[1]
        self.resolution_ = mapparams["resolution"]  # [meters] side of 1 grid square
        self.origin = np.array(mapparams["origin"][:2]).astype(np.float32)  # [meters] x y coordinates of point at i = j = 0
        if mapparams["origin"][2] != 0:
            raise ValueError(
                "Map files with a rotated frame (origin.theta != 0) are not"
                " supported. Setting the value to 0 in the MAP_NAME.yaml file is one way to"
                " resolve this."
            )
        self._thresh_occupied = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]
        self.HUGE_ = 100 * self.occupancy_shape0 * self.occupancy_shape1 # bigger than any possible distance in the map
        if self.resolution_ == 0:
            raise ValueError("resolution can not be 0")

    def cset_resolution(self, float res):
        self.resolution_ = res

    def set_resolution(self, res):
        self.cset_resolution(res)

    def resolution(self):
        res = float(self.resolution_)
        return res

    def thresh_occupied(self):
        res = float(self._thresh_occupied)
        return res

    def as_occupied_points_ij(self):
        return np.ascontiguousarray(np.array(np.where(self.occupancy() > self.thresh_occupied())).T)

    def as_closed_obst_vertices(self):
        """ Converts map into list of contours of obstacles, in xy
        returns list of obstacles, for each obstacle a list of xy vertices constituing its contour
        based on the opencv2 findContours function
        """

        cont = self.as_closed_obst_vertices_ij()
        # switch i j
        contours = [self.ij_to_xy(c) for c in cont]
        return contours

    def as_closed_obst_vertices_ij(self):
        """ Converts map into list of contours of obstacles, in ij
        returns list of obstacles, for each obstacle a list of ij vertices constituing its contour
        based on the opencv2 findContours function
        """
        import cv2
        gray = self.occupancy()
        ret, thresh = cv2.threshold(gray, self.thresh_occupied(), 1, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        cv2_output = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # len(cv2_output) depends on cv2 version :/
        if cv2.__version__[0] == '4':
            cont = cv2_output[0] 
        elif cv2.__version__[0] == '3':
            cont = cv2_output[1]
        else:
            raise NotImplementedError("cv version {} unsupported".format(cv2.__version__))
        # remove extra dim
        contours = [np.vstack([c[:,0,1], c[:,0,0]]).T for c in cont]
        return contours

    def plot_contours(self, *args, **kwargs):
        from matplotlib import pyplot as plt
        if not args:
            raise ValueError("args empty. contours must be supplied as first argument.")
        if len(args) == 1:
            args = args + ('-,',)
        contours = args[0]
        args = args[1:]
        for c in contours:
            # add the first vertice at the end to close the plotted contour
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            plt.plot(cplus[:,0], cplus[:,1], *args, **kwargs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cas_sdf(self, np.int64_t[:,::1] occupied_points_ij, np.float32_t[:, ::1] min_distances):
        """ everything in ij units """
        cdef np.int64_t[:] point
        cdef np.int64_t pi
        cdef np.int64_t pj
        cdef np.float32_t norm
        cdef np.int64_t i
        cdef np.int64_t j 
        cdef np.float32_t smallest_dist
        cdef int n_occupied_points_ij = len(occupied_points_ij)
        for i in range(min_distances.shape[0]):
            for j in range(min_distances.shape[1]):
                smallest_dist = min_distances[i, j]
                for k in range(n_occupied_points_ij):
                    point = occupied_points_ij[k]
                    pi = point[0]
                    pj = point[1]
                    norm = csqrt((pi - i) ** 2 + (pj - j) ** 2)
                    if norm < smallest_dist:
                        smallest_dist = norm
                min_distances[i, j] = smallest_dist

    def distance_transform_2d(self):
        f = np.zeros_like(self.occupancy(), dtype=np.float32)
        f[self.occupancy() <= self.thresh_occupied()] = np.inf
        D = np.ones_like(self.occupancy(), dtype=np.float32) * np.inf
        cdistance_transform_2d(f, D)
        return np.sqrt(D)*self.resolution()

    def distance_transform_2d_ij(self):
        f = np.zeros_like(self.occupancy(), dtype=np.float32)
        f[self.occupancy() <= self.thresh_occupied()] = np.inf
        D = np.ones_like(self.occupancy(), dtype=np.float32) * np.inf
        cdistance_transform_2d(f, D)
        return np.sqrt(D)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cas_tsdf(self, np.float32_t max_dist_m, np.int64_t[:,::1] occupied_points_ij, np.float32_t[:, ::1] min_distances):
        # DEPRECATED
        """ everything in ij units """
        cdef np.int64_t max_dist_ij = np.int64((max_dist_m / self.resolution_))
        cdef np.int64_t[:] point
        cdef np.int64_t pi
        cdef np.int64_t pj
        cdef np.float32_t norm
        cdef np.int64_t i
        cdef np.int64_t j 
        cdef np.int64_t iend
        cdef np.int64_t jend 

        for k in range(len(occupied_points_ij)):
            point = occupied_points_ij[k]
            pi = point[0]
            pj = point[1]
            i = max(pi - max_dist_ij, 0)
            iend = min(pi + max_dist_ij, min_distances.shape[0] - 1)
            j = max(pj - max_dist_ij, 0)
            jend = min(pj + max_dist_ij, min_distances.shape[1] - 1)
            while True:
                j = max(pj - max_dist_ij, 0)
                while True:
                    norm = csqrt((pi - i) ** 2 + (pj - j) ** 2)
                    if norm < min_distances[i, j]:
                        min_distances[i, j] = norm
                    j = j+1
                    if j >= jend: break
                i = i+1
                if i >= iend: break

    def as_tsdf(self, max_dist_m):
        # this is faster than the still poorly optimized cas_tsdf
        min_distances = self.as_sdf()
        min_distances[min_distances > max_dist_m] = max_dist_m
        min_distances[min_distances < -max_dist_m] = -max_dist_m
        return min_distances

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cxy_to_ij(self, np.float32_t[:,::1] xy, np.float32_t[:,::1] ij, bool clip_if_outside=True):
        if xy.shape[1] != 2:
            raise IndexError("xy should be of shape (n, 2)")
        for k in range(xy.shape[0]):
            ij[k, 0] = (xy[k, 0] - self.origin[0]) / self.resolution_
            ij[k, 1] = (xy[k, 1] - self.origin[1]) / self.resolution_
        if clip_if_outside:
            for k in range(xy.shape[0]):
                if ij[k, 0] >= self.occupancy_shape0:
                    ij[k, 0] = self.occupancy_shape0 - 1
                if ij[k, 1] >= self.occupancy_shape1:
                    ij[k, 1] = self.occupancy_shape1 - 1
                if ij[k, 0] < 0:
                    ij[k, 0] = 0
                if ij[k, 1] < 0:
                    ij[k, 1] = 0
        return ij

    def xy_to_ij(self, xy, clip_if_outside=True):
        if type(xy) is not np.ndarray:
            xy = np.array(xy)
        ij = np.zeros_like(xy, dtype=np.float32)
        self.cxy_to_ij(xy.astype(np.float32), ij, clip_if_outside)
        return ij.astype(np.int64)

    def xy_to_floatij(self, xy, clip_if_outside=True):
        if type(xy) is not np.ndarray:
            xy = np.array(xy)
        ij = np.zeros_like(xy, dtype=np.float32)
        self.cxy_to_ij(xy.astype(np.float32), ij, clip_if_outside)
        return ij

    def old_xy_to_ij(self, x, y=None, clip_if_outside=True):
        # if no y argument is given, assume x is a [...,2] array with xy in last dim
        """
        for each x y coordinate, return an i j cell index
        Examples
        --------
        >>> a = Map2D()
        >>> a.xy_to_ij(0.01, 0.02)
        (1, 2)
        >>> a.xy_to_ij([0.01, 0.02])
        array([1, 2])
        >>> a.xy_to_ij([[0.01, 0.02], [-0.01, 0.]])
        array([[1, 2],
               [0, 0]])
        """
        if y is None:
            return np.concatenate(
                self.xy_to_ij(
                    *np.split(np.array(x), 2, axis=-1), clip_if_outside=clip_if_outside
                ),
                axis=-1,
            )
        i = (x - self.origin[0]) / self.resolution_
        j = (y - self.origin[1]) / self.resolution_
        i = i.astype(int)
        j = j.astype(int)
        if clip_if_outside:
            i_gt = i >= self._occupancy.shape[0]
            i_lt = i < 0
            j_gt = j >= self._occupancy.shape[1]
            j_lt = j < 0
            if isinstance(i, np.ndarray):
                i[i_gt] = self._occupancy.shape[0] - 1
                i[i_lt] = 0
                j[j_gt] = self._occupancy.shape[1] - 1
                j[j_lt] = 0
            else:
                if i_gt:
                    i = self._occupancy.shape[0] - 1
                if i_lt:
                    i = 0
                if j_gt:
                    j = self._occupancy.shape[1] - 1
                if j_lt:
                    j = 0
        return i, j

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cij_to_xy(self, np.float32_t[:,::1] ij):
        xy = np.zeros([ij.shape[0], ij.shape[1]], dtype=np.float32)
        for k in range(ij.shape[0]):
            # adds 0.5 so that x y is in the middle of the cell. Otherwise ij->xy->ij is not identity
            xy[k, 0] = (ij[k, 0]+0.5) * self.resolution_ + self.origin[0]
            xy[k, 1] = (ij[k, 1]+0.5) * self.resolution_ + self.origin[1]
        return xy

    def ij_to_xy(self, i, j=None):
        """
        Examples
        --------
        >>> a = Map2D()
        >>> a.ij_to_xy(1, 2)
        (0.01, 0.02)
        >>> a.ij_to_xy([1,2])
        array([0.01, 0.02])
        >>> a.ij_to_xy([[1,2], [-1, 0]])
        array([[ 0.01,  0.02],
               [-0.01,  0.  ]])
        """
        # if no j argument is given, assume i is a [...,2] array with ij in last dim
        if j is None:
            return np.concatenate(
                self.ij_to_xy(*np.split(np.array(i), 2, axis=-1)), axis=-1
            )
        # adds 0.5 so that x y is in the middle of the cell. Otherwise ij->xy->ij is not identity
        x = (i + 0.5) * self.resolution_ + self.origin[0]
        y = (j + 0.5) * self.resolution_ + self.origin[1]
        return x, y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cis_inside_ij(self, np.float32_t[:,::1] ij):
        inside = np.ones([ij.shape[0],], dtype=np.bool)
        for k in range(ij.shape[0]):
            if ij[k, 0] >= self.occupancy_shape0:
                inside[k] = False
            if ij[k, 1] >= self.occupancy_shape1:
                inside[k] = False
            if ij[k, 0] < 0:
                inside[k] = False
            if ij[k, 1] < 0:
                inside[k] = False
        return inside

    def is_inside_ij(self, i, j=None):
        from functools import reduce
        """
        Examples
        --------
        >>> a = Map2D()
        >>> a.is_inside_ij(1, 2)
        True
        >>> a.is_inside_ij([1,2])
        array(True)
        >>> a.is_inside_ij([[1,2]])
        array([ True])
        >>> a.is_inside_ij([[1,a._occupancy.shape[1]]])
        array([False])
        >>> a.is_inside_ij([[a._occupancy.shape[0],2]])
        array([False])
        >>> a.is_inside_ij([[1,2], [-1, 0]])
        array([ True, False])
        """
        if j is None:
            return self.is_inside_ij(*np.split(np.array(i), 2, axis=-1))[..., 0]
        return reduce(
            np.logical_and,
            [i >= 0, i < self._occupancy.shape[0], j >= 0, j < self._occupancy.shape[1]],
        )

    def occupancy(self):
        occ = np.array(self._occupancy)
        return occ

    def occupancy_T(self):
        occ_T = np.zeros((self.occupancy_shape1, self.occupancy_shape0), dtype=np.float32)
        for i in range(self.occupancy_shape1):
            for j in range(self.occupancy_shape0):
                occ_T[i, j] = self._occupancy[j, i]
        return occ_T

    def as_sdf(self, raytracer=None):
        min_distances = self.distance_transform_2d()
        # Switch sign for occupied and unkown points (*signed* distance field)
        min_distances[self.occupancy() > self.thresh_free] *= -1.
        return min_distances

    def as_sdf_ij(self, raytracer=None):
        min_distances = self.distance_transform_2d_ij()
        # Switch sign for occupied and unkown points (*signed* distance field)
        min_distances[self.occupancy() > self.thresh_free] *= -1.
        return min_distances

    cpdef as_coarse_map2d(self, n=1):
        # recursion to provide a convenient way to coarsen x times
        if n > 1:
            return self.as_coarse_map2d(n=int(n-1)).as_coarse_map2d()
        coarse = CMap2D()
        # if the number of rows/column is not even, this will discard the last one
        coarse.occupancy_shape0 = int(cfloor(self.occupancy_shape0 / 2))
        coarse.occupancy_shape1 = int(cfloor(self.occupancy_shape1 / 2))
        coarse._occupancy = np.zeros((coarse.occupancy_shape0, coarse.occupancy_shape1), dtype=np.float32)
        for i in range(coarse.occupancy_shape0):
            for j in range(coarse.occupancy_shape1):
                coarse._occupancy[i, j] = max(
                        self._occupancy[i*2  , j*2  ],
                        self._occupancy[i*2+1, j*2  ],
                        self._occupancy[i*2  , j*2+1],
                        self._occupancy[i*2+1, j*2+1],
                        )

        coarse.resolution_ = self.resolution_ * 2
        coarse.origin = np.array([0., 0.], dtype=np.float32)
        coarse.origin[0] = self.origin[0]
        coarse.origin[1] = self.origin[1]
        coarse._thresh_occupied = self._thresh_occupied
        coarse.thresh_free = self.thresh_free
        coarse.HUGE_ = self.HUGE_
        return coarse

    def fastmarch(self, goal_ij, mask=None, speeds=None):
        """ 

        Nodes are cells in a 2d grid

        calculates time to goal (sec) , assuming speed at nodes (ij/sec)

        """
        # Mask (close) unattainable nodes
        if mask is None:
            mask = (self.occupancy() >= self.thresh_free).astype(np.uint8)
        # initialize extra costs
        if speeds is None:
            speeds = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.float32)
        # initialize field to large value
        inv_value = np.inf
        result = np.ones_like(self.occupancy(), dtype=np.float32) * inv_value
        if not self.is_inside_ij(goal_ij[0], goal_ij[1]):
            raise ValueError("Goal ij ({}, {}) not inside map of size ({}, {})".format(
                goal_ij[0], goal_ij[1], self.occupancy_shape0, self.occupancy_shape1))
        self.cfastmarch(goal_ij, result, mask, speeds)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cfastmarch(self, np.int64_t[:] goal_ij,
            np.float32_t[:, ::1] tentative,
            np.uint8_t[:, ::1] mask,
            np.float32_t[:, ::1] speeds,
            ):
        # Initialize bool arrays
        cdef np.uint8_t[:, ::1] open_ = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.uint8)
        # Mask (close) unattainable nodes
        for i in range(self.occupancy_shape0):
            for j in range(self.occupancy_shape1):
                if mask[i, j]:
                    open_[i, j] = 0
        # Start at the goal location
        tentative[goal_ij[0], goal_ij[1]] = 0
        cdef cpp_priority_queue[cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]]] priority_queue
        priority_queue.push(
                cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]](0, cpp_pair[np.int64_t, np.int64_t](goal_ij[0], goal_ij[1]))
                )
        cdef cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]] popped
        cdef np.int64_t popped_idxi
        cdef np.int64_t popped_idxj
        cdef np.int64_t[:, ::1] neighbor_offsets
        neighbor_offsets = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64) # first row must be up right down left
        cdef np.int64_t n_neighbor_offsets = len(neighbor_offsets)
        cdef np.int64_t len_i = tentative.shape[0]
        cdef np.int64_t len_j = tentative.shape[1]
        cdef np.int64_t smallest_tentative_id
        cdef np.float32_t value
        cdef np.float32_t smallest_tentative_value
        cdef np.int64_t node_idxi
        cdef np.int64_t node_idxj
        cdef np.int64_t neighbor_idxi
        cdef np.int64_t neighbor_idxj
        cdef np.int64_t oi
        cdef np.int64_t oj
        cdef np.int64_t currenti = goal_ij[0]
        cdef np.int64_t currentj = goal_ij[1]
        cdef np.float32_t new_cost
        cdef np.float32_t old_cost
        cdef np.float32_t a
        cdef np.float32_t b
        cdef np.float32_t s
        cdef np.float32_t s2inv
        cdef np.float32_t delta
        while not priority_queue.empty():
            # Pop the node with the smallest tentative value from the to_visit list
            while not priority_queue.empty():
                popped = priority_queue.top()
                priority_queue.pop()
                popped_idxi = popped.second.first
                popped_idxj = popped.second.second
                # skip nodes which are already closed (stagnant duplicates in the heap)
                if open_[popped_idxi, popped_idxj] == 1:
                    currenti = popped_idxi
                    currentj = popped_idxj
                    break
            # Iterate over neighbors
            for n in range(n_neighbor_offsets):
                # Indices for the neighbours
                oi = neighbor_offsets[n, 0]
                oj = neighbor_offsets[n, 1]
                neighbor_idxi = currenti + oi
                neighbor_idxj = currentj + oj
                # exclude forbidden/explored areas of the grid
                if neighbor_idxi < 0:
                    continue
                if neighbor_idxi >= len_i:
                    continue
                if neighbor_idxj < 0:
                    continue
                if neighbor_idxj >= len_j:
                    continue
                # Exclude invalid neighbors
                if not open_[neighbor_idxi, neighbor_idxj]:
                    continue
                # Fastmarch update
                a = np.inf
                if neighbor_idxi != 0:
                    a = tentative[neighbor_idxi-1, neighbor_idxj]
                if neighbor_idxi != len_i-1:
                    a = min(a, tentative[neighbor_idxi+1, neighbor_idxj])
                b = np.inf
                if neighbor_idxj != 0:
                    b = tentative[neighbor_idxi, neighbor_idxj-1]
                if neighbor_idxj != len_j-1:
                    b = min(b, tentative[neighbor_idxi, neighbor_idxj+1])
                s = speeds[neighbor_idxi, neighbor_idxj]
                s2inv = 1./s**2
                delta = 2 * s2inv - (a-b)**2
                if delta > 0:
                    new_cost = ( a + b + csqrt(delta) ) / 2
                else:
                    new_cost = 1./s + min(a,b)
                old_cost = tentative[neighbor_idxi, neighbor_idxj]
                if new_cost < old_cost or old_cost == np.inf:
                    tentative[neighbor_idxi, neighbor_idxj] = new_cost
                    # Add neighbor to priority queue
                    priority_queue.push(
                            cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]](
                                -new_cost, cpp_pair[np.int64_t, np.int64_t](neighbor_idxi, neighbor_idxj))
                            )
            # Close the current node
            open_[currenti, currentj] = 0
        return tentative

    def dijkstra(self, goal_ij, mask=None, extra_costs=None, inv_value=None, connectedness=8):
        """ 4, 8, 16, or 32 connected dijkstra 

        Nodes are cells in a 2d grid
        Assumes edge costs are xy distance between two nodes

        """
        # Mask (close) unattainable nodes
        if mask is None:
            mask = (self.occupancy() >= self.thresh_free).astype(np.uint8)
        # initialize extra costs
        if extra_costs is None:
            extra_costs = np.zeros((self.occupancy_shape0, self.occupancy_shape1), dtype=np.float32)
        # initialize field to large value
        if inv_value is None:
            inv_value = self.HUGE_
        result = np.ones_like(self.occupancy(), dtype=np.float32) * inv_value
        if not self.is_inside_ij(goal_ij[0], goal_ij[1]):
            raise ValueError("Goal ij ({}, {}) not inside map of size ({}, {})".format(
                goal_ij[0], goal_ij[1], self.occupancy_shape0, self.occupancy_shape1))
        self.cdijkstra(goal_ij, result, mask, extra_costs, inv_value, connectedness)
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cdijkstra(self, np.int64_t[:] goal_ij,
            np.float32_t[:, ::1] tentative,
            np.uint8_t[:, ::1] mask,
            np.float32_t[:, ::1] extra_costs,
            np.float32_t inv_value, connectedness=8):
        cdef np.float32_t kEdgeLength = 1. * self.resolution_  # meters
        # Initialize bool arrays
        cdef np.uint8_t[:, ::1] open_ = np.ones((self.occupancy_shape0, self.occupancy_shape1), dtype=np.uint8)
        # Mask (close) unattainable nodes
        for i in range(self.occupancy_shape0):
            for j in range(self.occupancy_shape1):
                if mask[i, j]:
                    open_[i, j] = 0
        # Start at the goal location
        tentative[goal_ij[0], goal_ij[1]] = 0
        cdef cpp_priority_queue[cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]]] priority_queue
        priority_queue.push(
                cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]](0, cpp_pair[np.int64_t, np.int64_t](goal_ij[0], goal_ij[1]))
                )
        cdef cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]] popped
        cdef np.int64_t popped_idxi
        cdef np.int64_t popped_idxj
        cdef np.int64_t[:, ::1] neighbor_offsets
        if connectedness == 32:
            neighbor_offsets = np.array([
                [0, 1], [ 1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
                [1, 1], [ 1,-1], [-1, 1], [-1,-1],
                [2, 1], [ 2,-1], [-2, 1], [-2,-1],
                [1, 2], [-1, 2], [ 1,-2], [-1,-2],
                [3, 1], [ 3,-1], [-3, 1], [-3,-1],
                [1, 3], [-1, 3], [ 1,-3], [-1,-3],
                [3, 2], [ 3,-2], [-3, 2], [-3,-2],
                [2, 3], [-2, 3], [ 2,-3], [-2,-3]], dtype=np.int64)
        elif connectedness==16:
            neighbor_offsets = np.array([
                [0, 1], [ 1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
                [1, 1], [ 1,-1], [-1, 1], [-1,-1],
                [2, 1], [ 2,-1], [-2, 1], [-2,-1],
                [1, 2], [-1, 2], [ 1,-2], [-1,-2]], dtype=np.int64)
        elif connectedness==8:
            neighbor_offsets = np.array([
                [0, 1], [1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
                [1, 1], [1,-1], [-1, 1], [-1,-1]], dtype=np.int64)
        elif connectedness==4:
            neighbor_offsets = np.array([
                [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64) # first row must be up right down left
        else:
            raise ValueError("invalid value {} for connectedness passed as argument".format(connectedness))
        cdef np.int64_t n_neighbor_offsets = len(neighbor_offsets)
        cdef np.int64_t len_i = tentative.shape[0]
        cdef np.int64_t len_j = tentative.shape[1]
        cdef np.int64_t smallest_tentative_id
        cdef np.float32_t value
        cdef np.float32_t smallest_tentative_value
        cdef np.int64_t node_idxi
        cdef np.int64_t node_idxj
        cdef np.int64_t neighbor_idxi
        cdef np.int64_t neighbor_idxj
        cdef np.int64_t oi
        cdef np.int64_t oj
        cdef np.int64_t currenti = goal_ij[0]
        cdef np.int64_t currentj = goal_ij[1]
        cdef np.float32_t edge_extra_costs
        cdef np.float32_t new_cost
        cdef np.float32_t old_cost
        cdef np.float32_t edge_ratio
        cdef np.uint8_t[::1] blocked = np.zeros((8), dtype=np.uint8)
        while not priority_queue.empty():
            # Pop the node with the smallest tentative value from the to_visit list
            while not priority_queue.empty():
                popped = priority_queue.top()
                priority_queue.pop()
                popped_idxi = popped.second.first
                popped_idxj = popped.second.second
                # skip nodes which are already closed (stagnant duplicates in the heap)
                if open_[popped_idxi, popped_idxj] == 1:
                    currenti = popped_idxi
                    currentj = popped_idxj
                    break
            # Iterate over neighbors
            for n in range(n_neighbor_offsets):
                # Indices for the neighbours
                oi = neighbor_offsets[n, 0]
                oj = neighbor_offsets[n, 1]
                neighbor_idxi = currenti + oi
                neighbor_idxj = currentj + oj
                edge_ratio = csqrt(oi**2 + oj**2)
                # exclude forbidden/explored areas of the grid
                if neighbor_idxi < 0:
                    continue
                if neighbor_idxi >= len_i:
                    continue
                if neighbor_idxj < 0:
                    continue
                if neighbor_idxj >= len_j:
                    continue
                # check whether path is obstructed (for 16/32 connectedness)
                if n < 4:
                    blocked[n] = mask[neighbor_idxi, neighbor_idxj]
                elif n < 8:
                    blocked[n] = mask[neighbor_idxi, neighbor_idxj]
                # Exclude obstructed jumps (for 16/32 connectedness)
                if n > 4: # for example, prevent ur if u is blocked
                    # assumes first row of offsets is up right down left (see offset init!)
                    if (oj > 0 and blocked[0]) or \
                       (oi > 0 and blocked[1]) or \
                       (oj < 0 and blocked[2]) or \
                       (oi < 0 and blocked[3]):
                           continue
                if n > 8: # for example, prevent uuur if ur is blocked
                    # assumes second row ru rd lu ld
                    if (oi > 0 and oj > 0 and blocked[4]) or \
                       (oi > 0 and oj < 0 and blocked[5]) or \
                       (oi < 0 and oj > 0 and blocked[6]) or \
                       (oi < 0 and oj < 0 and blocked[7]):
                           continue
                # Exclude invalid neighbors
                if not open_[neighbor_idxi, neighbor_idxj]:
                    continue
                # costly regions are expensive to navigate through (costlier edges)
                # these extra costs have to be reciprocal in order for dijkstra to function
                # cost(a to b) == cost(b to a), hence the average between the node penalty values.
                # Find which neighbors are open (exclude forbidden/explored areas of the grid)
                edge_extra_costs = 0.5 * (
                    extra_costs[neighbor_idxi, neighbor_idxj]
                    + extra_costs[currenti, currentj]
                )
                new_cost = (
                    tentative[currenti, currentj] + kEdgeLength * edge_ratio + edge_extra_costs
                )
                old_cost = tentative[neighbor_idxi, neighbor_idxj]
                if new_cost < old_cost or old_cost == inv_value:
                    tentative[neighbor_idxi, neighbor_idxj] = new_cost
                    # Add neighbor to priority queue
                    priority_queue.push(
                            cpp_pair[np.float32_t, cpp_pair[np.int64_t, np.int64_t]](
                                -new_cost, cpp_pair[np.int64_t, np.int64_t](neighbor_idxi, neighbor_idxj))
                            )
            # Close the current node
            open_[currenti, currentj] = 0
        return tentative

    def old_render_agents_in_lidar(self, ranges, angles, agents, lidar_ij):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid """
        centers_i = [0]
        centers_j = [0]
        radii_ij = [np.inf]
        for agent in agents:
            if agent.type != "legs":
                raise NotImplementedError
            left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame = agent.get_legs_pose2d_in_map()
            llc_ij = self.xy_to_ij(left_leg_pose2d_in_map_frame[:2])
            rlc_ij = self.xy_to_ij(right_leg_pose2d_in_map_frame[:2])
            leg_radius_ij = agent.leg_radius / self.resolution_
            # circle centers in 'lidar' frame (frame centered at lidar pos, but not rotated,
            # as angles in array are already rotated according to sensor angle in map frame)
            centers_i.append(llc_ij[0] - lidar_ij[0])
            centers_j.append(llc_ij[1] - lidar_ij[1])
            radii_ij.append(leg_radius_ij)
            centers_i.append(rlc_ij[0] - lidar_ij[0])
            centers_j.append(rlc_ij[1] - lidar_ij[1])
            radii_ij.append(leg_radius_ij)
        # switch to polar coordinate to find intersection between each ray and agent (circles)
        angles = np.array(angles)
        ranges = np.array(ranges)
        radii_ij = np.array(radii_ij)
        centers_r_sq = np.array(centers_i)**2 + np.array(centers_j)**2
        centers_l = np.arctan2(centers_j, centers_i)
        # Circle in polar coord: r^2 - 2*r*r0*cos(phi-lambda) + r0^2 = R^2
        # Solve equation for r at angle phi in polar coordinates, of circle of center (r0, lambda)
        # and radius R. -> 2 solutions for r knowing r0, phi, lambda, R: 
        # r = r0*cos(phi-lambda) - sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # r = r0*cos(phi-lambda) + sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # solutions are real only if term inside sqrt is > 0
        first_term = np.sqrt(centers_r_sq) * np.cos(angles[:,None] - centers_l)
        sqrt_inner = centers_r_sq * np.cos(angles[:,None] - centers_l)**2 - centers_r_sq + radii_ij**2
        sqrt_inner[sqrt_inner < 0] = np.inf
        radii_solutions_a = first_term - np.sqrt(sqrt_inner)
        radii_solutions_b = first_term + np.sqrt(sqrt_inner)
        radii_solutions_a[radii_solutions_a < 0] = np.inf
        radii_solutions_b[radii_solutions_b < 0] = np.inf
        # range is the smallest range between original range, and intersection range with each agent
        all_sol = np.hstack([radii_solutions_a, radii_solutions_b])
        all_sol = all_sol * self.resolution_ # from ij coordinates back to xy
        all_sol[:,0] = ranges
        final_ranges = np.min(all_sol, axis=1)
        ranges[:] = final_ranges

    def render_agents_in_many_lidars(self, ranges, xythetas, agents):
        self.crender_agents_in_many_lidars(ranges, xythetas, agents)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef crender_agents_in_many_lidars(self,
            np.ndarray[np.float32_t, ndim=3, mode='c'] ranges,   # agent, points, n_lidars_p_agent
            np.float32_t[:,:,:,::1] ijthetas, # agent, points, n_lidars_p_agent, i j th
            agents,
            ):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid
        assumes the angles are ordered from lowest to highest, spaced evenly (const increment)
        """
        cdef np.float32_t PI = np.pi
        cdef int n_agents = len(agents)
        cdef int n_angles = ijthetas.shape[1]
        cdef int n_lidars_per_agent = ijthetas.shape[2]
        if n_agents == 0:
            return True
        cdef int n_centers = 2* (n_agents - 1) # 2 legs per agent, one less agent (excluded)
        cdef np.float32_t[:] centers_i = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_j = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] radii_ij = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_r_sq = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_l = np.zeros((n_centers,), dtype=np.float32)
        # loop variables
        cdef CSimAgent cagent
        cdef np.float32_t[:, ::1] left_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] right_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] llc_ij = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] rlc_ij = np.zeros((1,3), dtype=np.float32)
        cdef int i1 = 0
        cdef int i2 = 0
        cdef np.float32_t leg_radius_ij
        cdef np.float32_t[:, :, ::1] llcijs = np.zeros((n_agents, 1, 3), dtype=np.float32)
        cdef np.float32_t[:, :, ::1] rlcijs = np.zeros((n_agents, 1, 3), dtype=np.float32)
        cdef np.float32_t[:] leg_radii_ijs  = np.zeros((n_agents,), dtype=np.float32)
        for n in range(n_agents):
            agent = agents[n]
            cagent = CSimAgent(agent.pose_2d_in_map_frame, agent.state)
            cagent.cget_legs_pose2d_in_map(left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame)
            self.cxy_to_ij(left_leg_pose2d_in_map_frame[:1,:2], llc_ij)
            self.cxy_to_ij(right_leg_pose2d_in_map_frame[:1, :2], rlc_ij)
            leg_radii_ijs[n] = cagent.leg_radius / self.resolution_
            llcijs[n, 0, 0] = llc_ij[0, 0]
            llcijs[n, 0, 1] = llc_ij[0, 1]
            llcijs[n, 0, 2] = llc_ij[0, 2]
            rlcijs[n, 0, 0] = rlc_ij[0, 0]
            rlcijs[n, 0, 1] = rlc_ij[0, 1]
            rlcijs[n, 0, 2] = rlc_ij[0, 2]
        # final calculation cdefs
        cdef np.float32_t angle_min
        cdef np.float32_t angle_max
        cdef np.float32_t angle_inc
        cdef np.float32_t angle_0_ref
        cdef np.float32_t r0sq
        cdef np.float32_t r0
        cdef np.float32_t lmbda
        cdef np.float32_t R
        cdef np.float32_t phimin
        cdef np.float32_t phimax
        cdef np.float32_t phi
        cdef np.float32_t first_term
        cdef np.float32_t sqrt_inner
        cdef np.float32_t min_solution
        cdef np.float32_t possible_solution
        cdef np.float32_t possible_solution_m
        cdef int indexmin
        cdef int indexmax
        cdef int k
        cdef bool wholescan = False
        for a in range(n_agents): # apply to each agent
            for lr in range(n_lidars_per_agent): # apply to left / right lidar
                # apply render agents to single lidar scan
                k = 0
                for o_a in range(n_agents):
                    # fetch leg pos for other agents (except current)
                    if o_a == a:
                        continue
                    leg_radius_ij = leg_radii_ijs[o_a]
                    llc_ij[0, 0] = llcijs[o_a, 0, 0]
                    llc_ij[0, 1] = llcijs[o_a, 0, 1]
                    llc_ij[0, 2] = llcijs[o_a, 0, 2]
                    rlc_ij[0, 0] = rlcijs[o_a, 0, 0]
                    rlc_ij[0, 1] = rlcijs[o_a, 0, 1]
                    rlc_ij[0, 2] = rlcijs[o_a, 0, 2]
                    # circle centers in 'lidar' frame (frame centered at lidar pos, but not rotated,
                    # as angles in array are already rotated according to sensor angle in map frame)
                    i1 = 2*k # even index, for left leg
                    i2 = 2*k+1 # odd index for right leg
                    centers_i[i1] = llc_ij[0, 0] - ijthetas[a, 0, lr, 0]
                    centers_j[i1] = llc_ij[0, 1] - ijthetas[a, 0, lr, 1]
                    radii_ij[i1] = leg_radius_ij
                    centers_i[i2] = rlc_ij[0, 0] - ijthetas[a, 0, lr, 0]
                    centers_j[i2] = rlc_ij[0, 1] - ijthetas[a, 0, lr, 1]
                    radii_ij[i2] = leg_radius_ij
                    # switch to polar coordinate to find intersection between each ray and agent (circles)
                    centers_r_sq[i1] = centers_i[i1]**2 + centers_j[i1]**2
                    centers_l[i1] = np.arctan2(centers_j[i1], centers_i[i1])
                    centers_r_sq[i2] = centers_i[i2]**2 + centers_j[i2]**2
                    centers_l[i2] = np.arctan2(centers_j[i2], centers_i[i2])
                    k += 1
                # Circle in polar coord: r^2 - 2*r*r0*cos(phi-lambda) + r0^2 = R^2
                # Solve equation for r at angle phi in polar coordinates, of circle of center (r0, lambda)
                # and radius R. -> 2 solutions for r knowing r0, phi, lambda, R: 
                # r = r0*cos(phi-lambda) - sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
                # r = r0*cos(phi-lambda) + sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
                # solutions are real only if term inside sqrt is > 0
                angle_min = ijthetas[a, 0, lr, 2]
                angle_max = ijthetas[a, n_angles-1, lr, 2]
                angle_inc = ijthetas[a, 1, lr, 2] - angle_min
                if angle_min >= angle_max:
                    raise ValueError("angles expected to be ordered from min to max.")
                # angle_0_ref is a multiple of 2pi, the closest one smaller than angles[0]
                # assuming a scan covers less than full circle, all angles in the scan should lie 
                # between angle_0_ref and angle_0_ref + 2* 2pi (two full circles)
                angle_0_ref = 2*PI * (angle_min // (2*PI))
                for i in range(n_centers):
                    r0sq = centers_r_sq[i]
                    r0 = csqrt(r0sq)
                    lmbda = centers_l[i]
                    R = radii_ij[i]
                    # we can first check at what angles this holds.
                    # there should be two extrema for the circle in phi, which are solutions for:
                    # r0^2*cos^2(phi-lambda) - r0^2 + R^2 = 0 
                    # the two solutions are:
                    # phi = lambda + 2*pi*n +- arccos( +- sqrt(r0^2 - R^2) / r0 )
                    # these exist only if r0 > R and r0 != 0
                    if centers_r_sq[i] == 0 or centers_r_sq[i] < radii_ij[i]**2:
                        indexmin = 0
                        indexmax = n_angles - 1
                    else:
                        phimin = lmbda - cacos( csqrt(r0sq - R**2) / r0 )
                        phimax = lmbda + cacos( csqrt(r0sq - R**2) / r0 )
                        #                    this is phi as an angle [0, 2pi]
                        phimin = angle_0_ref + phimin % (PI * 2)
                        phimax = angle_0_ref + phimax % (PI * 2)
                        # try the second full circle if our agent is outside the scan
                        if phimax < angle_min:
                            phimin = phimin + PI * 2
                            phimax = phimax + PI * 2
                        # if still outside the scan, our agent is not visible
                        if phimax < angle_min or phimin > angle_max:
                            continue
                        # find the index for the first visible circle point in the scan
                        indexmin = int( ( max(phimin, angle_min) - angle_min ) // angle_inc )
                        indexmax = int( ( min(phimax, angle_max) - angle_min ) // angle_inc )
                    for idx in range(indexmin, indexmax+1):
                        phi = ijthetas[a, idx, lr, 2]
                        first_term = r0 * ccos(phi - lmbda)
                        sqrt_inner = r0sq * ccos(phi - lmbda)**2 - r0sq + R**2
                        if sqrt_inner < 0:
                            # in this case that ray does not see the agent
                            continue
                        min_solution = ranges[a, idx, lr] # initialize with scan range
                        possible_solution = first_term - csqrt(sqrt_inner) # in ij units
                        possible_solution_m = possible_solution * self.resolution_ # in meters
                        if possible_solution_m >= 0:
                            min_solution = min(min_solution, possible_solution_m)
                        possible_solution = first_term + csqrt(sqrt_inner)
                        possible_solution_m = possible_solution * self.resolution_
                        if possible_solution_m >= 0:
                            min_solution = min(min_solution, possible_solution_m)
                        ranges[a, idx, lr] = min_solution
        return True

    def render_agents_in_lidar(self, ranges, angles, agents, lidar_ij):
        if not self.crender_agents_in_lidar(ranges, angles.astype(np.float32), agents, lidar_ij.astype(np.float32)):
#             print("in rendering agents, object too close for efficient solution")
            self.old_render_agents_in_lidar(ranges, angles, agents, lidar_ij)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef crender_agents_in_lidar(self,
            np.ndarray[np.float32_t, ndim=1] ranges,
            np.ndarray[np.float32_t, ndim=1] angles,
            agents,
            np.ndarray[np.float32_t, ndim=1] lidar_ij,
            ):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid
        assumes the angles are ordered from lowest to highest, spaced evenly (const increment)
        """
        if len(agents) == 0:
            return True
        cdef int n_centers = 2* len(agents)
        cdef np.float32_t[:] centers_i = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_j = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] radii_ij = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_r_sq = np.zeros((n_centers,), dtype=np.float32)
        cdef np.float32_t[:] centers_l = np.zeros((n_centers,), dtype=np.float32)
        # loop variables
        cdef CSimAgent cagent
        cdef np.float32_t[:, ::1] left_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] right_leg_pose2d_in_map_frame = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] llc_ij = np.zeros((1,3), dtype=np.float32)
        cdef np.float32_t[:, ::1] rlc_ij = np.zeros((1,3), dtype=np.float32)
        cdef int i1 = 0
        cdef int i2 = 0
        cdef np.float32_t leg_radius_ij
        for n in range(len(agents)):
            agent = agents[n]
            cagent = CSimAgent(agent.pose_2d_in_map_frame, agent.state)
            cagent.cget_legs_pose2d_in_map(left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame)
            self.cxy_to_ij(left_leg_pose2d_in_map_frame[:1,:2], llc_ij)
            self.cxy_to_ij(right_leg_pose2d_in_map_frame[:1, :2], rlc_ij)
            leg_radius_ij = cagent.leg_radius / self.resolution_
            # circle centers in 'lidar' frame (frame centered at lidar pos, but not rotated,
            # as angles in array are already rotated according to sensor angle in map frame)
            i1 = 2*n # even index, for left leg
            i2 = 2*n+1 # odd index for right leg
            centers_i[i1] = llc_ij[0, 0] - lidar_ij[0]
            centers_j[i1] = llc_ij[0, 1] - lidar_ij[1]
            radii_ij[i1] = leg_radius_ij
            centers_i[i2] = rlc_ij[0, 0] - lidar_ij[0]
            centers_j[i2] = rlc_ij[0, 1] - lidar_ij[1]
            radii_ij[i2] = leg_radius_ij
            # switch to polar coordinate to find intersection between each ray and agent (circles)
            centers_r_sq[i1] = centers_i[i1]**2 + centers_j[i1]**2
            centers_l[i1] = np.arctan2(centers_j[i1], centers_i[i1])
            centers_r_sq[i2] = centers_i[i2]**2 + centers_j[i2]**2
            centers_l[i2] = np.arctan2(centers_j[i2], centers_i[i2])
        # Circle in polar coord: r^2 - 2*r*r0*cos(phi-lambda) + r0^2 = R^2
        # Solve equation for r at angle phi in polar coordinates, of circle of center (r0, lambda)
        # and radius R. -> 2 solutions for r knowing r0, phi, lambda, R: 
        # r = r0*cos(phi-lambda) - sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # r = r0*cos(phi-lambda) + sqrt( r0^2*cos^2(phi-lambda) - r0^2 + R^2 )
        # solutions are real only if term inside sqrt is > 0
        for i in range(n_centers):
            # if an object is too close, this will not be efficient, tell the caller to switch to numpy
            if centers_r_sq[i] == 0 or centers_r_sq[i] < radii_ij[i]**2:
                return False
        # we can first check at what angles this holds.
        # there should be two extrema for the circle in phi, which are solutions for:
        # r0^2*cos^2(phi-lambda) - r0^2 + R^2 = 0 
        # the two solutions are:
        # phi = lambda + 2*pi*n +- arccos( +- sqrt(r0^2 - R^2) / r0 )
        # these exist only if r0 > R and r0 != 0
        cdef np.float32_t angle_min = angles[0]
        cdef np.float32_t angle_max = angles[len(angles)-1]
        cdef np.float32_t angle_inc = angles[1] - angle_min
        if angle_min >= angle_max:
            raise ValueError("angles expected to be ordered from min to max.")
        # angle_0_ref is a multiple of 2pi, the closest one smaller than angles[0]
        # assuming a scan covers less than full circle, all angles in the scan should lie 
        # between angle_0_ref and angle_0_ref + 2* 2pi (two full circles)
        cdef np.float32_t angle_0_ref = 2*np.pi * (angle_min // (2*np.pi))
        # loop variables
        cdef np.float32_t r0sq
        cdef np.float32_t r0
        cdef np.float32_t lmbda
        cdef np.float32_t R
        cdef np.float32_t phimin
        cdef np.float32_t phimax
        cdef np.float32_t phi
        cdef np.float32_t first_term
        cdef np.float32_t sqrt_inner
        cdef np.float32_t min_solution
        cdef np.float32_t possible_solution
        cdef np.float32_t possible_solution_m
        cdef int indexmin
        cdef int indexmax
        for i in range(n_centers):
            r0sq = centers_r_sq[i]
            r0 = np.sqrt(r0sq)
            lmbda = centers_l[i]
            R = radii_ij[i]
            phimin = lmbda - np.arccos( np.sqrt(r0sq - R**2) / r0 )
            phimax = lmbda + np.arccos( np.sqrt(r0sq - R**2) / r0 )
            #                    this is phi as an angle [0, 2pi]
            phimin = angle_0_ref + phimin % (np.pi * 2)
            phimax = angle_0_ref + phimax % (np.pi * 2)
            # try the second full circle if our agent is outside the scan
            if phimax < angle_min:
                phimin = phimin + np.pi * 2
                phimax = phimax + np.pi * 2
            # if still outside the scan, our agent is not visible
            if phimax < angle_min or phimin > angle_max:
                continue
            # find the index for the first visible circle point in the scan
            indexmin = int( ( max(phimin, angle_min) - angle_min ) // angle_inc )
            indexmax = int( ( min(phimax, angle_max) - angle_min ) // angle_inc )
            for idx in range(indexmin, indexmax+1):
                phi = angles[idx]
                first_term = r0 * np.cos(phi - lmbda)
                sqrt_inner = r0sq * np.cos(phi - lmbda)**2 - r0sq + R**2
                if sqrt_inner < 0:
                    # in this case that ray does not see the agent
                    continue
                min_solution = ranges[idx] # initialize with scan range
                possible_solution = first_term - np.sqrt(sqrt_inner) # in ij units
                possible_solution_m = possible_solution * self.resolution_ # in meters
                if possible_solution_m >= 0:
                    min_solution = min(min_solution, possible_solution_m)
                possible_solution = first_term + np.sqrt(sqrt_inner)
                possible_solution_m = possible_solution * self.resolution_
                if possible_solution_m >= 0:
                    min_solution = min(min_solution, possible_solution_m)
                ranges[idx] = min_solution
        return True

    def visibility_map(self, observer_ij):
        return self.visibility_map_ij(observer_ij) * self.resolution()

    def visibility_map_ij(self, observer_ij):
        visibility_map = np.ones_like(self.occupancy(), dtype=np.float32) * -1
        self.cvisibility_map_ij(np.array(observer_ij).astype(np.int64), visibility_map)
        return visibility_map

    cdef cvisibility_map_ij(self, np.int64_t[::1] observer_ij, np.float32_t[:, ::1] visibility_map):
        cdef np.int64_t o_i = observer_ij[0]
        cdef np.int64_t o_j = observer_ij[1]
        cdef np.float32_t threshold = self._thresh_occupied
        cdef np.int64_t shape0 = self.occupancy_shape0
        cdef np.int64_t shape1 = self.occupancy_shape1
        max_r = np.maximum(shape0, shape1)
        cdef np.float32_t min_angle_increment = 1. / max_r
        cdef np.float32_t angle = 0.
        cdef np.float32_t TWOPI = np.pi * 2.
        cdef np.float32_t i_inc_unit
        cdef np.float32_t j_inc_unit
        cdef np.float32_t i_abs_inc
        cdef np.float32_t j_abs_inc
        cdef np.float32_t raystretch
        cdef np.int64_t max_inc
        cdef np.int64_t max_i_inc
        cdef np.int64_t max_j_inc
        cdef np.float32_t i_inc
        cdef np.float32_t j_inc
        cdef np.float32_t n_i
        cdef np.float32_t n_j
        cdef np.int64_t in_i
        cdef np.int64_t in_j
        cdef np.float32_t occ
        cdef np.int64_t di
        cdef np.int64_t dj
        cdef np.float32_t r
        cdef np.uint8_t is_hit
        while True:
            angle_increment = 0
            if angle >= TWOPI:
                break
            i_inc_unit = ccos(angle)
            j_inc_unit = csin(angle)
            # Stretch the ray so that every 1 unit in the ray direction lands on a cell in i or 
            i_abs_inc = abs(i_inc_unit)
            j_abs_inc = abs(j_inc_unit)
            raystretch = 1. / i_abs_inc if i_abs_inc >= j_abs_inc else 1. / j_abs_inc
            i_inc = i_inc_unit * raystretch
            j_inc = j_inc_unit * raystretch
            # max amount of increments before crossing the grid border
            if i_inc == 0:
                max_inc = <np.int64_t>((shape1 - 1 - o_j) / j_inc) if j_inc >= 0 else <np.int64_t>(o_j / -j_inc)
            elif j_inc == 0:
                max_inc = <np.int64_t>((shape0 - 1 - o_i) / i_inc) if i_inc >= 0 else <np.int64_t>(o_i / -i_inc)
            else:
                max_i_inc = <np.int64_t>((shape1 - 1 - o_j) / j_inc) if j_inc >= 0 else <np.int64_t>(o_j / -j_inc)
                max_j_inc = <np.int64_t>((shape0 - 1 - o_i) / i_inc) if i_inc >= 0 else <np.int64_t>(o_i / -i_inc)
                max_inc = max_i_inc if max_i_inc <= max_j_inc else max_j_inc
            # Trace a ray
            n_i = o_i + 0
            n_j = o_j + 0
            for n in range(1, max_inc-1):
                n_i += i_inc
                in_i = <np.int64_t>n_i
                in_j = <np.int64_t>n_j
                di = ( in_i - o_i )
                dj = ( in_j - o_j )
                r = sqrt(di*di + dj*dj)
                visibility_map[in_i, in_j] = r
                if r != 0:
                    occ = self._occupancy[in_i, in_j]
                    if occ >= threshold:
                        angle_increment = 0.99 / r
                        angle += angle_increment
                        break
                n_j += j_inc
                in_i = <np.int64_t>n_i
                in_j = <np.int64_t>n_j
                di = ( in_i - o_i )
                dj = ( in_j - o_j )
                r = sqrt(di*di + dj*dj)
                visibility_map[in_i, in_j] = r
                if r != 0:
                    occ = self._occupancy[in_i, in_j]
                    if occ >= threshold:
                        angle_increment = 0.99 / r
                        angle += angle_increment
                        break
            # if we hit the edge of the map
            if angle_increment == 0:
                angle_increment = min_angle_increment
                angle += angle_increment

    cdef craymarch(self, np.int64_t[::1] observer_ij, np.int64_t[::1] angles, np.float32_t[::1] ranges):
        cdef np.int64_t o_i = observer_ij[0]
        cdef np.int64_t o_j = observer_ij[1]
        cdef np.float32_t[:,::1] esdf_ij = self.as_sdf_ij()
        cdef np.float32_t resolution = self.resolution()
        cdef np.float32_t threshold = self._thresh_occupied
        cdef np.int64_t shape0 = self.occupancy_shape0
        cdef np.int64_t shape1 = self.occupancy_shape1
        cdef np.float32_t angle = 0.
        cdef np.float32_t i_inc_unit
        cdef np.float32_t j_inc_unit
        cdef np.float32_t i_abs_inc
        cdef np.float32_t j_abs_inc
        cdef np.float32_t raystretch
        cdef np.int64_t max_inc
        cdef np.int64_t max_i_inc
        cdef np.int64_t max_j_inc
        cdef np.float32_t i_inc
        cdef np.float32_t j_inc
        cdef np.float32_t n_i
        cdef np.float32_t n_j
        cdef np.int64_t in_i
        cdef np.int64_t in_j
        cdef np.float32_t occ
        cdef np.int64_t di
        cdef np.int64_t dj
        cdef np.float32_t r
        for k in range(angles.shape[0]):
            angle = angles[k]
            i_inc_unit = ccos(angle)
            j_inc_unit = csin(angle)
            # Stretch the ray so that every 1 unit in the ray direction lands on a cell in i or 
            i_abs_inc = abs(i_inc_unit)
            j_abs_inc = abs(j_inc_unit)
            raystretch = 1. / i_abs_inc if i_abs_inc >= j_abs_inc else 1. / j_abs_inc
            i_inc = i_inc_unit * raystretch
            j_inc = j_inc_unit * raystretch
            # max amount of increments before crossing the grid border
            if i_inc == 0:
                max_inc = <np.int64_t>((shape1 - 1 - o_j) / j_inc) if j_inc >= 0 else <np.int64_t>(o_j / -j_inc)
            elif j_inc == 0:
                max_inc = <np.int64_t>((shape0 - 1 - o_i) / i_inc) if i_inc >= 0 else <np.int64_t>(o_i / -i_inc)
            else:
                max_i_inc = <np.int64_t>((shape1 - 1 - o_j) / j_inc) if j_inc >= 0 else <np.int64_t>(o_j / -j_inc)
                max_j_inc = <np.int64_t>((shape0 - 1 - o_i) / i_inc) if i_inc >= 0 else <np.int64_t>(o_i / -i_inc)
                max_inc = max_i_inc if max_i_inc <= max_j_inc else max_j_inc
            # Trace a ray
            n_i = o_i + 0
            n_j = o_j + 0
            in_i = <np.int64_t>n_i
            in_j = <np.int64_t>n_j
            for n in range(1, max_inc-1):
                closest = esdf_ij[in_i, in_j]
                if closest > 5:
                    n_i += i_inc_unit * (closest - 1)
                    n_j += j_inc_unit * (closest - 1)

                occ3 = self._occupancy[in_i, <np.int64_t>(n_j+j_inc)] # makes the beam 'thicker' by checking intermediate pixels
                n_i += i_inc
                in_i = <np.int64_t>n_i
                occ2 = self._occupancy[in_i, in_j]
                n_j += j_inc
                in_j = <np.int64_t>n_j
                occ = self._occupancy[in_i, in_j]
                if occ >= threshold or occ2 >= threshold or occ3 >= threshold:
                    di = ( in_i - o_i )
                    dj = ( in_j - o_j )
                    r = sqrt(di*di + dj*dj) * resolution
                    ranges[k] = r
                    break

cdef class CSimAgent:
    cdef public np.float32_t[:] pose_2d_in_map_frame
    cdef public str type
    cdef public np.float32_t[:] state
    cdef public float leg_radius

    def __cinit__(self, pose, state):
        self.pose_2d_in_map_frame = pose
        self.type = "legs"
        self.state = state
        self.leg_radius = 0.03 # [m]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef cget_legs_pose2d_in_map(self,
            np.float32_t[:, ::1] left_leg_pose2d_in_map_frame, 
            np.float32_t[:, ::1] right_leg_pose2d_in_map_frame):
        if self.type != "legs":
            raise NotImplementedError
        cdef np.float32_t[:] m_a_T = self.pose_2d_in_map_frame
        cdef np.float32_t leg_radius = self.leg_radius # [m]
        cdef np.float32_t leg_side_offset = 0.1 # [m]
        cdef np.float32_t leg_side_amplitude = 0.1 # [m] half amplitude
        cdef np.float32_t leg_front_amplitude = 0.3 # [m]
        # get position of each leg w.r.t agent (x is 'forward')
        # travel is a sine function relative to how fast the agent is moving in x y
        cdef np.float32_t front_travel =  leg_front_amplitude * ccos(
                self.state[0] * 2. / leg_front_amplitude # assuming dx = 2 dphi / A
                + self.state[2] # adds a little movement when rotating
                )
        cdef np.float32_t side_travel =  leg_side_amplitude * ccos(
                self.state[1] * 2. / leg_side_amplitude
                + self.state[2]
                )
        cdef np.float32_t[:, ::1] right_leg_pose2d_in_agent_frame = np.zeros((1,3), dtype=np.float32)
        right_leg_pose2d_in_agent_frame[0, 0] = front_travel
        right_leg_pose2d_in_agent_frame[0, 1] = side_travel + leg_side_offset
        right_leg_pose2d_in_agent_frame[0, 2] = 0
        cdef np.float32_t[:, ::1] left_leg_pose2d_in_agent_frame = np.zeros((1,3), dtype=np.float32) 
        left_leg_pose2d_in_agent_frame[0, 0] = -1 * right_leg_pose2d_in_agent_frame[0, 0]
        left_leg_pose2d_in_agent_frame[0, 1] = -1 * right_leg_pose2d_in_agent_frame[0, 1]
        left_leg_pose2d_in_agent_frame[0, 2] = -1 * right_leg_pose2d_in_agent_frame[0, 2]
        capply_tf_to_pose(
                left_leg_pose2d_in_agent_frame, m_a_T, left_leg_pose2d_in_map_frame)
        capply_tf_to_pose(
                right_leg_pose2d_in_agent_frame, m_a_T, right_leg_pose2d_in_map_frame)

    def get_legs_pose2d_in_map(self):
        m_a_T = self.pose_2d_in_map_frame
        if self.type == "legs":
            leg_radius = self.leg_radius # [m]
            leg_side_offset = 0.1 # [m]
            leg_side_amplitude = 0.1 # [m] half amplitude
            leg_front_amplitude = 0.3 # [m]
            # get position of each leg w.r.t agent (x is 'forward')
            # travel is a sine function relative to how fast the agent is moving in x y
            front_travel =  leg_front_amplitude * np.cos(
                    self.state[0] * 2. / leg_front_amplitude # assuming dx = 2 dphi / A
                    + self.state[2] # adds a little movement when rotating
                    )
            side_travel =  leg_side_amplitude * np.cos(
                    self.state[1] * 2. / leg_side_amplitude
                    + self.state[2]
                    )
            right_leg_pose2d_in_agent_frame = np.array([
                front_travel,
                side_travel + leg_side_offset,
                0])
            left_leg_pose2d_in_agent_frame =  -right_leg_pose2d_in_agent_frame
            left_leg_pose2d_in_map_frame = apply_tf_to_pose(
                    left_leg_pose2d_in_agent_frame, m_a_T)
            right_leg_pose2d_in_map_frame = apply_tf_to_pose(
                    right_leg_pose2d_in_agent_frame, m_a_T)
            return left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame
        else:
            raise NotImplementedError




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cas_sdf(self, np.int64_t[:,::1] occupied_points_ij, np.float32_t[:, ::1] min_distances):
    """ everything in ij units """
    cdef np.int64_t[:] point
    cdef np.int64_t pi
    cdef np.int64_t pj
    cdef np.float32_t norm
    cdef np.int64_t i
    cdef np.int64_t j 
    cdef np.float32_t smallest_dist
    cdef int n_occupied_points_ij = len(occupied_points_ij)
    for i in range(min_distances.shape[0]):
        for j in range(min_distances.shape[1]):
            smallest_dist = min_distances[i, j]
            for k in range(n_occupied_points_ij):
                point = occupied_points_ij[k]
                pi = point[0]
                pj = point[1]
                norm = csqrt((pi - i) ** 2 + (pj - j) ** 2)
                if norm < smallest_dist:
                    smallest_dist = norm
            min_distances[i, j] = smallest_dist

cdef cdistance_transform_1d(np.float32_t[::1] f, np.float32_t[::1] D):
    """ based on 'Distance Transforms of Sampled Functions' by Felzenswalb et al """
    if f.shape[0] != D.shape[0]:
        raise IndexError
    cdef np.float32_t[::1] z = np.zeros((f.shape[0]+1), dtype=np.float32) # z[i] intersection between lowest parabola i and lowest parabola i-1
    cdef np.int64_t[::1] v = np.zeros((f.shape[0]), dtype=np.int64) # the positions of parabolas forming the lower envelope
    cdef np.int64_t k  = 0 # the amount of parabolas forming the lower envelope (- 1 for indexing)
    cdef np.int64_t max_k  = 0 # the amount of parabolas forming the lower envelope (- 1 for indexing)
    cdef np.int64_t start = 0 
    cdef np.int64_t vk = 0 # var for speed
    z[0] = -np.inf # boundary with previous
    z[1] = np.inf # boundary with next
    # find and add the first non-inf parabola to the lower envelope
    for q in range(0, f.shape[0]):
        if f[q] == np.inf:
            continue
        v[0] = q
        z[0] = -np.inf # boundary with previous
        z[1] = np.inf # boundary with next
        break
    start = v[k] + 1 # start after the first non-inf parabola
    for q in range(start, f.shape[0]):
        if f[q] == np.inf: # inf parabolas are too 'high' to affect lower envelope
            continue
        while True:
            vk = v[k]
            s = ((f[q] + q*q) - (f[vk] + vk*vk)) / (2*q - 2*vk) # boundary with previous parabola
            if s <= z[k]:
                # the boundary between this and the previous parabola is before the boundary between the previous and its predecessor
                # the latest parabola obsoletes the previous one, erase the previous one from the L.E
                k = k-1
            elif s > z[k]:
                # normal situation, add this parabola to the L.E and continue
                k = k+1
                v[k] = q
                z[k] = s
                z[k + 1] = np.inf
                break
    max_k = k
    k = 0
    for q in range(f.shape[0]):
        # find the parabola corresponding to the current L.E section (where z[k] < q < z[k+1])
        # move k forward until the boundary with next is later than q
        while z[k+1] < q:
            k = k+1
        vk = v[k]
        D[q] = f[vk] + (q - vk)**2

cdef cdistance_transform_2d(np.float32_t[:, ::1] f, np.float32_t[:, ::1] D):
    cdef np.float32_t[:, ::1] f_T = np.ascontiguousarray(np.copy(f).T)
    cdef np.float32_t[:, ::1] D_after_vertical_pass_T = np.ascontiguousarray(np.copy(D).T)
    cdef np.float32_t[:, ::1] D_after_vertical_pass
    # vertical pass
    for j in range(f.shape[1]):
        cdistance_transform_1d(f_T[j,:] , D_after_vertical_pass_T[j,:])
    D_after_vertical_pass = np.ascontiguousarray(np.copy(D_after_vertical_pass_T).T)
    # vertical pass
    # horizontal pass
    for i in range(f.shape[0]):
        cdistance_transform_1d(D_after_vertical_pass[i,:] , D[i,:])




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef capply_tf_to_pose(np.float32_t[:, ::1] pose, np.float32_t[:] pose2d,
    np.float32_t[:, ::1] result):
    cdef np.float32_t th = pose2d[2]
    result[0, 0] = ccos(th) * pose[0, 0] - csin(th) * pose[0, 1] + pose2d[0]
    result[0, 1] = csin(th) * pose[0, 0] + ccos(th) * pose[0, 1] + pose2d[1]
    result[0, 2] = pose[0, 2] + th

def apply_tf(x, pose2d):
    # x is in frame B
    # pose2d is AT_B
    # result is x in frame A
    return rotate(x, pose2d[2]) + np.array(pose2d[:2])

def apply_tf_to_pose(pose, pose2d):
    # same as apply_tf but assumes pose is x y theta instead of x y
    xy = rotate(np.array([pose[:2]]), pose2d[2])[0] + np.array(pose2d[:2])
    th = pose[2] + pose2d[2]
    return np.array([xy[0], xy[1], th])

def apply_tf_to_vel(vel, pose2d):
    # same as apply_tf but assumes vel is xdot ydot thetadot instead of x y
    # for xdot ydot frame transformation applies, 
    # but thetadot is invariant due to frames being fixed.
    xy = rotate(np.array([vel[...,:2]]), pose2d[2])[0]
    th = vel[...,2]
    return np.array([xy[0], xy[1], th])

def rotate(x, th):
    rotmat = np.array([
        [np.cos(th), -np.sin(th)],
        [np.sin(th), np.cos(th)],
        ])
    return np.matmul(rotmat, x.T).T

def inverse_pose2d(pose2d):
    inv_th = -pose2d[2] # theta
    inv_xy = rotate(np.array([-pose2d[:2]]), inv_th)[0]
    return np.array([inv_xy[0], inv_xy[1], inv_th])


def path_from_dijkstra_field(costmap, first, connectedness=8):
    """ returns a path in ij coordinates based on a costmap and an initial position """
    return cpath_from_dijkstra_field(costmap,
            np.array(first).astype(np.int64),
            connectedness=connectedness)

cdef cpath_from_dijkstra_field(np.float32_t[:,::1] costmap, np.int64_t[::1] first, connectedness=8):
    # 8 connected
    # Neighbor offsets
    cdef np.int64_t[:,::1] offsets
    if connectedness == 32:
        offsets = np.array([
            [0, 1], [ 1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
            [1, 1], [ 1,-1], [-1, 1], [-1,-1], # second row must be ru rd lu ld
            [2, 1], [ 2,-1], [-2, 1], [-2,-1],
            [1, 2], [-1, 2], [ 1,-2], [-1,-2],
            [3, 1], [ 3,-1], [-3, 1], [-3,-1],
            [1, 3], [-1, 3], [ 1,-3], [-1,-3],
            [3, 2], [ 3,-2], [-3, 2], [-3,-2],
            [2, 3], [-2, 3], [ 2,-3], [-2,-3]], dtype=np.int64)
    elif connectedness==16:
        offsets = np.array([
            [0, 1], [ 1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
            [1, 1], [ 1,-1], [-1, 1], [-1,-1],
            [2, 1], [ 2,-1], [-2, 1], [-2,-1],
            [1, 2], [-1, 2], [ 1,-2], [-1,-2]], dtype=np.int64)
    elif connectedness==8:
        offsets = np.array([
            [0, 1], [1, 0], [ 0,-1], [-1, 0], # first row must be up right down left
            [1, 1], [1,-1], [-1, 1], [-1,-1]], dtype=np.int64)
    elif connectedness==4:
        offsets = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64) # first row must be up right down left
    else:
        raise ValueError("invalid value {} for connectedness passed as argument".format(connectedness))
    # Init
    path = []
    jump_log = []
    path.append([first[0], first[1]])
    cdef np.int64_t n_offsets = len(offsets)
    cdef np.int64_t maxi = costmap.shape[0]
    cdef np.int64_t maxj = costmap.shape[1]
    cdef np.int64_t current_idxi = first[0]
    cdef np.int64_t current_idxj = first[1]
    cdef np.float32_t current_cost
    cdef np.int64_t offset_idxi
    cdef np.int64_t offset_idxj
    cdef np.int64_t oi
    cdef np.int64_t oj
    cdef np.float32_t olen
    cdef np.float32_t[::1] offset_edge_costs = np.zeros((n_offsets,), dtype=np.float32)
    cdef np.float32_t offset_edge_cost
    cdef np.float32_t best_offset_edge_cost
    cdef np.int64_t n_best_edges
    cdef np.int64_t[::1] tied_firstplace_candidates = np.zeros((n_offsets,), dtype=np.int64)
    cdef np.int64_t stochastic_candidate_pick
    cdef np.uint8_t[::1] blocked = np.zeros((8), dtype=np.uint8)
    # Path in global lowres map ij frame
    while True:
        current_cost = costmap[current_idxi, current_idxj]
        # lookup all edge costs and find lowest cost which is also < 0
        best_offset_edge_cost = 0
        for n in range(n_offsets):
            oi = offsets[n, 0]
            oj = offsets[n, 1]
            offset_idxi = current_idxi + oi
            offset_idxj = current_idxj + oj
            if offset_idxi < 0 or offset_idxi >= maxi or offset_idxj < 0 or offset_idxj >= maxj:
                offset_edge_cost = 0
            else:
                offset_edge_cost = costmap[offset_idxi, offset_idxj] - current_cost
                # check whether path is obstructed (for 16/32 connectedness)
                if n < 4:
                    if offset_edge_cost >= 0:
                        blocked[n] = 1
                elif n < 8:
                    if offset_edge_cost >= 0:
                        blocked[n] = 1
                # Exclude obstructed jumps (for 16/32 connectedness)
                if n > 4: # for example, prevent ur if u is blocked
                    # assumes first row of offsets is up right down left (see offset init!)
                    if (oj > 0 and blocked[0]) or \
                       (oi > 0 and blocked[1]) or \
                       (oj < 0 and blocked[2]) or \
                       (oi < 0 and blocked[3]):
                           offset_edge_cost = 0
                if n > 8: # for example, prevent uuur if ur is blocked
                    # second row ru rd lu ld
                    if (oi > 0 and oj > 0 and blocked[4]) or \
                       (oi > 0 and oj < 0 and blocked[5]) or \
                       (oi < 0 and oj > 0 and blocked[6]) or \
                       (oi < 0 and oj < 0 and blocked[7]):
                           offset_edge_cost = 0
            # in 8/16 connectedness some offsets are 'longer' and will be preferred unless normalized
            olen = csqrt(oi*oi + oj*oj)
            offset_edge_cost = offset_edge_cost / olen
            # store for later
            offset_edge_costs[n] = offset_edge_cost
            if offset_edge_cost < best_offset_edge_cost:
                best_offset_edge_cost = offset_edge_cost
        # find how many choice are tied for best cost, if several, sample stochastically
        n_best_edges = 0
        if best_offset_edge_cost >= 0:
            # local minima reached, terminate
            jump_log.append(n_best_edges)
            break
        for n in range(n_offsets):
            if offset_edge_costs[n] == best_offset_edge_cost:
                tied_firstplace_candidates[n_best_edges] = n
                n_best_edges += 1
        if n_best_edges > 1:
            # probabilistic jump (pick between best candidates)
            stochastic_candidate_pick = np.random.randint(n_best_edges, dtype=np.int64)
            selected_offset_id = tied_firstplace_candidates[stochastic_candidate_pick]
        elif n_best_edges == 1:
            selected_offset_id = tied_firstplace_candidates[0]
        jump_log.append(n_best_edges)
        current_idxi = current_idxi + offsets[selected_offset_id, 0]
        current_idxj = current_idxj + offsets[selected_offset_id, 1]
        path.append([current_idxi, current_idxj])
    return np.array(path), np.array(jump_log)
