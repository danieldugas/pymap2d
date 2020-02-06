from __future__ import print_function
from matplotlib.pyplot import imread
from functools import partial
import numpy as np
import os
import threading
from yaml import load
from circular_index import CircularIndexCreator
import pose2d

from numba import njit
from math import sqrt, floor, log


@njit(fastmath=True, cache=True)
def compiled_occupied_points(occupancy, thresh):
    result = []
    for i in range(occupancy.shape[0]):
        for j in range(occupancy.shape[1]):
            if occupancy[i, j] > thresh:
                result.append([i, j])
    return result


# in i_j_ frame
@njit(fastmath=True, cache=True)
def compiled_sdf_math(occupied_points_ij, init):
    for i in range(init.shape[0]):
        for j in range(init.shape[1]):
            minnorm = init[i, j]
            for k in range(len(occupied_points_ij)):
                point = occupied_points_ij[k]
                pi = point[0]
                pj = point[1]
                norm = sqrt((pi - i) ** 2 + (pj - j) ** 2)
                minnorm = minnorm if minnorm < norm else norm
            init[i, j] = minnorm


# in i_j_ frame
@njit(fastmath=True, cache=True)
def compiled_tsdf_math(occupied_points_ij, init, max_dist_ij):
    for k in range(len(occupied_points_ij)):
        point = occupied_points_ij[k]
        pi = point[0]
        pj = point[1]
        irange = range(
            max(pi - max_dist_ij, 0), min(pi + max_dist_ij, init.shape[0] - 1)
        )
        jrange = range(
            max(pj - max_dist_ij, 0), min(pj + max_dist_ij, init.shape[1] - 1)
        )
        for i in irange:
            for j in jrange:
                norm = sqrt((pi - i) ** 2 + (pj - j) ** 2)
                if norm < init[i, j]:
                    init[i, j] = norm


@njit(fastmath=True, cache=True)
def compiled_dijkstra(
    open_, not_in_to_visit, kEdgeLength, extra_costs, tentative, goal_ij
):
    tentative = tentative[:]
    tentative[goal_ij[0], goal_ij[1]] = 0
    to_visit = [(goal_ij[0], goal_ij[1])]
    not_in_to_visit[goal_ij[0], goal_ij[1]] = False
    neighbor_offsets = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    len_i = tentative.shape[0]
    len_j = tentative.shape[1]
    while to_visit:
        # Make the current node that which has the smallest tentative values
        smallest_tentative_value = tentative[to_visit[0][0], to_visit[0][1]]
        smallest_tentative_id = 0
        for i in range(len(to_visit)):
            node_idx = to_visit[i]
            value = tentative[node_idx[0], node_idx[1]]
            if value < smallest_tentative_value:
                smallest_tentative_value = value
                smallest_tentative_id = i
        current = to_visit.pop(smallest_tentative_id)
        # Iterate over 4 neighbors
        for n in range(4):
            # Indices for the neighbours
            neighbor_idx = (
                current[0] + neighbor_offsets[n][0],
                current[1] + neighbor_offsets[n][1],
            )
            # Find which neighbors are open (exclude forbidden/explored areas of the grid)
            if neighbor_idx[0] < 0:
                continue
            if neighbor_idx[0] >= len_i:
                continue
            if neighbor_idx[1] < 0:
                continue
            if neighbor_idx[1] >= len_j:
                continue
            if not open_[neighbor_idx[0], neighbor_idx[1]]:
                continue
            # costly regions are expensive to navigate through (costlier edges)
            # these extra costs have to be reciprocal in order for dijkstra to function
            # cost(a to b) == cost(b to a), hence the average between the node penalty values.
            # Find which neighbors are open (exclude forbidden/explored areas of the grid)
            edge_extra_costs = 0.5 * (
                extra_costs[neighbor_idx[0], neighbor_idx[1]]
                + extra_costs[current[0], current[1]]
            )
            new_cost = (
                tentative[current[0], current[1]] + kEdgeLength + edge_extra_costs
            )
            if new_cost < tentative[neighbor_idx[0], neighbor_idx[1]]:
                tentative[neighbor_idx[0], neighbor_idx[1]] = new_cost
            # Add neighbors to to_visit if not already present
            if not_in_to_visit[neighbor_idx[0], neighbor_idx[1]]:
                to_visit.append(neighbor_idx)
                not_in_to_visit[neighbor_idx[0], neighbor_idx[1]] = False
        # Close the current node
        open_[current[0], current[1]] = False
    return tentative


@njit(fastmath=True, cache=True)
def compiled_reverse_raytrace(ij_hits, ij_orig, init):
    o_i = ij_orig[0]
    o_j = ij_orig[1]
    for k in range(len(ij_hits)):
        ij = ij_hits[k]
        i = ij[0]
        j = ij[1]
        # length of the ray in increments
        d_i = i - o_i
        d_j = j - o_j
        sign_i = 1 if d_i >= 0 else -1
        sign_j = 1 if d_j >= 0 else -1
        i_len = sign_i * d_i
        j_len = sign_j * d_j
        if i_len >= j_len:
            ray_inc = i_len
        else:
            ray_inc = j_len
        if ray_inc == 0:
            continue
        # calculate increments
        i_inc = d_i * 1. / ray_inc
        j_inc = d_j * 1. / ray_inc
        # max amount of increments before crossing the grid border
        if i_inc == 0:
            max_inc = (
                floor((init.shape[1] - 1 - o_j) / j_inc)
                if sign_j >= 0
                else floor(o_j / -j_inc)
            )
        elif j_inc == 0:
            max_inc = (
                floor((init.shape[0] - 1 - o_i) / i_inc)
                if sign_i >= 0
                else floor(o_i / -i_inc)
            )
        else:
            max_i_inc = (
                floor((init.shape[0] - 1 - o_i) / i_inc)
                if sign_i >= 0
                else floor(o_i / -i_inc)
            )
            max_j_inc = (
                floor((init.shape[1] - 1 - o_j) / j_inc)
                if sign_j >= 0
                else floor(o_j / -j_inc)
            )
            max_inc = max_i_inc if max_i_inc <= max_j_inc else max_j_inc
        if ray_inc < max_inc:
            max_inc = ray_inc
        # Trace a ray
        n_i = o_i + 0
        n_j = o_j + 0
        for n in range(1, max_inc):
            init[
                int(n_i), int(n_j + j_inc)
            ] = 0  # the extra assignments make the ray 'thicker'
            n_i += i_inc
            init[int(n_i), int(n_j)] = 0
            n_j += j_inc
            init[int(n_i), int(n_j)] = 0
    # set occupancy to 1 for ray hits (after the rest to avoid erasing hits)
    for k in range(len(ij_hits)):
        ij = ij_hits[k]
        i = ij[0]
        j = ij[1]
        if (
            i > 0
            and j > 0
            and i != o_i
            and j != o_j
            and i < init.shape[0]
            and j < init.shape[1]
        ):
            init[i, j] = 1


@njit(fastmath=True, cache=True)
def compiled_insert_scan(hits_and_misses, prior, phit, pmiss):
    for i in range(hits_and_misses.shape[0]):
        for j in range(hits_and_misses.shape[1]):
            h_or_m = hits_and_misses[i, j]
            if h_or_m == 1.:  # hit
                pz = phit
            elif h_or_m == 0.:  # miss
                pz = pmiss
            else:
                continue
            pold = prior[i, j]
            # calculate posterior in-place
            prior[i, j] = pz * pold / (pz * pold + (1. - pz) * (1. - pold))
    return prior


class Map2D(object):
    def __init__(self, folder=None, name=None):
        print("WARNING: Map2D is deprecated in favor of CMap2D")
        self.occupancy_ = np.ones((100, 100)) * 0.5
        self.resolution_ = 0.01
        self.origin = np.array([0., 0.])
        self.thresh_occupied_ = 0.9
        self.thresh_free = 0.1
        self.HUGE = 1e10
        self.circular_index_creator = CircularIndexCreator() # Tool to quickly create gridmap circles
        if folder is None or name is None:
            return
        # Load map from file
        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")
        print("Loading map definition from {}".format(yaml_file))
        with open(yaml_file) as stream:
            mapparams = load(stream)
        map_file = os.path.join(folder, mapparams["image"])
        print("Map definition found. Loading map from {}".format(map_file))
        mapimage = imread(map_file)
        self.occupancy_ = (
            1. - mapimage.T[:, ::-1] / 254.
        )  # (0 to 1) 1 means 100% certain occupied
        self.resolution_ = mapparams["resolution"]  # [meters] side of 1 grid square
        self.origin = mapparams["origin"][
            :2
        ]  # [meters] x y coordinates of point at i = j = 0
        if mapparams["origin"][2] != 0:
            raise ValueError(
                "Map files with a rotated frame (origin.theta != 0) are not"
                " supported. Setting the value to 0 in the MAP_NAME.yaml file is one way to"
                " resolve this."
            )
        self.thresh_occupied_ = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]
        self.HUGE = 100 * np.prod(
            self.occupancy_.shape
        )  # bigger than any possible distance in the map

    def occupancy(self):
        return self.occupancy_

    def thresh_occupied(self):
        return self.thresh_occupied_

    def resolution(self):
        return self.resolution_

    def from_msg(self, msg):
        self.origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.resolution_ = msg.info.resolution
        self.occupancy_ = (
            np.array(msg.data).reshape((msg.info.height, msg.info.width)).T * 0.01
        )
        self.HUGE = 100 * np.prod(
            self.occupancy_.shape
        )  # bigger than any possible distance in the map

    def from_scan(self, scan, resolution, limits):
        """ Creating a map from a scan places the x y origin in the center of the grid,
        and generates the occupancy field from the laser data.
        """
        self.origin = limits[:, 0]
        self.resolution_ = resolution
        self.thresh_occupied_ = 0.9
        self.thresh_free = 0.1
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        xy_hits = (np.array(scan.ranges) * np.array([np.cos(angles), np.sin(angles)])).T
        ij_hits = self.xy_to_ij(xy_hits, clip_if_outside=False)
        self.occupancy_ = 0.5 * np.ones(
            ((limits[:, 1] - limits[:, 0]) / resolution).astype(int)
        )
        ij_laser_orig = (-self.origin / self.resolution_).astype(int)
        compiled_reverse_raytrace(ij_hits, ij_laser_orig, self.occupancy_)

    def xy_to_ij(self, x, y=None, clip_if_outside=True):
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
            i_gt = i >= self.occupancy_.shape[0]
            i_lt = i < 0
            j_gt = j >= self.occupancy_.shape[1]
            j_lt = j < 0
            if isinstance(i, np.ndarray):
                i[i_gt] = self.occupancy_.shape[0] - 1
                i[i_lt] = 0
                j[j_gt] = self.occupancy_.shape[1] - 1
                j[j_lt] = 0
            else:
                if i_gt:
                    i = self.occupancy_.shape[0] - 1
                if i_lt:
                    i = 0
                if j_gt:
                    j = self.occupancy_.shape[1] - 1
                if j_lt:
                    j = 0
        return i, j

    def xy_to_floatij(self, x):
        x = np.array(x)
        return (x - self.origin) / self.resolution_

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
        x = i * self.resolution_ + self.origin[0]
        y = j * self.resolution_ + self.origin[1]
        return x, y

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
        >>> a.is_inside_ij([[1,a.occupancy_.shape[1]]])
        array([False])
        >>> a.is_inside_ij([[a.occupancy_.shape[0],2]])
        array([False])
        >>> a.is_inside_ij([[1,2], [-1, 0]])
        array([ True, False])
        """
        if j is None:
            return self.is_inside_ij(*np.split(np.array(i), 2, axis=-1))[..., 0]
        return reduce(
            np.logical_and,
            [i > 0, i < self.occupancy_.shape[0], j > 0, j < self.occupancy_.shape[1]],
        )

    def as_square_graph(self):
        print()

    def square(size):
        rng = np.arange(-size, size)
        sz = np.ones(rng.shape) * size
        # indices along perimeter
        np.block([[rng, rng, sz, -sz], [sz, -sz, rng, rng]])
        # indices within square
        ii, jj = np.meshgrid(rng, rng)
        return np.stack((ii.flatten(), jj.flatten()), axis=-1).T

    def as_sdf(self, raytracer=None):
        NUMBA = False
        RANGE_LIBC = True
        occupied_points_ij = np.array(self.as_occupied_points_ij())
        min_distances = np.ones(self.occupancy_.shape) * self.HUGE
        if NUMBA:
            compiled_sdf_math(occupied_points_ij, min_distances)
        if RANGE_LIBC:
            if raytracer is None:
                import range_libc
                pyomap = range_libc.PyOMap(np.ascontiguousarray(self.occupancy_.T >= self.thresh_occupied_))
                rm = range_libc.PyRayMarching(pyomap, self.occupancy_.shape[0])
                min_distances = np.ascontiguousarray(np.zeros_like(self.occupancy_, dtype=np.float32))
                rm.get_dist(min_distances)
            else:
                min_distances = raytracer.get_dist()
        # Change from i, j units to x, y units [meters]
        min_distances = min_distances * self.resolution_
        # Switch sign for occupied and unkown points (*signed* distance field)
        min_distances[self.occupancy_ > self.thresh_free] *= -1.
        return min_distances

    def as_tsdf(self, max_dist_m):
        max_dist_ij = max_dist_m / self.resolution_
        occupied_points_ij = np.array(self.as_occupied_points_ij())
        min_distances = np.ones(self.occupancy_.shape) * max_dist_ij
        compiled_tsdf_math(occupied_points_ij, min_distances, max_dist_ij)
        # Change from i, j units to x, y units [meters]
        min_distances = min_distances * self.resolution_
        # Switch sign for occupied and unkown points (*signed* distance field)
        min_distances[self.occupancy_ > self.thresh_free] *= -1.
        return min_distances

    def as_occupied_points_ij(self, numba=False):
        if numba:
            return compiled_occupied_points(self.occupancy_, self.thresh_occupied_)
        return np.array(np.where(self.occupancy_ > self.thresh_occupied_)).T

    def as_coarse_map2d(self):
        from copy import deepcopy

        coarse = deepcopy(self)
        coarse.occupancy_ = np.max(
            np.stack(
                (
                    self.occupancy_[::2, ::2],
                    self.occupancy_[1::2, ::2],
                    self.occupancy_[::2, 1::2],
                    self.occupancy_[1::2, 1::2],
                ),
                axis=-1,
            ),
            axis=-1,
        )
        coarse.resolution_ = self.resolution_ * 2
        return coarse

    def dijkstra_deprecated(self, goal_ij, mask=None, extra_costs=None):
        kEdgeLength = 1 * self.resolution_  # meters
        # Initialize bool arrays
        open_ = np.ones(self.occupancy_.shape, dtype=bool)
        not_in_to_visit = np.ones(self.occupancy_.shape, dtype=bool)
        # Mask (close) unattainable nodes
        if mask is None:
            mask = self.occupancy_ >= self.thresh_free
        open_[mask] = False
        # initialize extra costs
        if extra_costs is None:
            extra_costs = 0
        # initialize field to large value
        tentative = np.ones(self.occupancy_.shape) * self.HUGE
        # Start at the goal location
        tentative[tuple(goal_ij)] = 0
        to_visit = [goal_ij]
        not_in_to_visit[tuple(goal_ij)] = False
        while to_visit:
            # Make the current node that which has the smallest tentative values
            smallest_tentative = np.argmin(tentative[tuple(np.array(to_visit).T)])
            current = to_visit.pop(smallest_tentative)
            # Indices for the neighbours
            neighbors = current + np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
            # costly regions are expensive to navigate through (costlier edges)
            # these extra costs have to be reciprocal in order for dijkstra to function
            # cost(a to b) == cost(b to a), hence the average between the node penalty values.
            edge_extra_costs = 0.5 * (
                extra_costs[tuple(neighbors.T)] + extra_costs[tuple(current)]
            )
            # Find which neighbors are open (exclude forbidden/explored areas of the grid)
            neighbors_are_open = open_[tuple(neighbors.T)]
            open_neighbors_idx = neighbors[neighbors_are_open]
            # Update neighbor values if smaller
            tentative[tuple(open_neighbors_idx.T)] = np.minimum(
                tentative[tuple(current)]
                + kEdgeLength
                + edge_extra_costs[neighbors_are_open],
                tentative[tuple(open_neighbors_idx.T)],
            )
            # Close the current node
            open_[tuple(current)] = False
            # Add current node neighbors to to_visit
            new_open_neighbors = open_neighbors_idx[
                not_in_to_visit[tuple(open_neighbors_idx.T)]
            ]
            not_in_to_visit[tuple(new_open_neighbors.T)] = False
            to_visit += list(new_open_neighbors)
        return tentative

    def dijkstra(self, goal_ij, mask=None, extra_costs=None):
        kEdgeLength = 1 * self.resolution_  # meters
        # Initialize bool arrays
        open_ = np.ones(self.occupancy_.shape, dtype=bool)
        not_in_to_visit = np.ones(self.occupancy_.shape, dtype=bool)
        # Mask (close) unattainable nodes
        if mask is None:
            mask = self.occupancy_ >= self.thresh_free
        open_[mask] = False
        # initialize extra costs
        if extra_costs is None:
            extra_costs = np.zeros_like(self.occupancy_)
        # initialize field to large value
        tentative = np.ones(self.occupancy_.shape) * self.HUGE
        # Start at the goal location
        tentative[tuple(goal_ij)] = 0
        to_visit = [goal_ij]
        not_in_to_visit[tuple(goal_ij)] = False
        return compiled_dijkstra(
            open_, not_in_to_visit, kEdgeLength, extra_costs, tentative, goal_ij
        )

    def as_closed_obst_vertices(self):
        # convert map into clusters
        # 
        import cv2
        im = cv2.imread('test.jpg')
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        pass

    def render_agents_in_lidar(self, ranges, angles, agents, lidar_ij):
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

    def render_agents_in_occupancy(self, agents):
        """ Takes a list of agents (shapes + position) and renders them into the occupancy grid """
        result_occupancy_grid = self.occupancy_.copy()
        for agent in agents:
            if agent.type != "legs":
                raise NotImplementedError
            left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame = agent.get_legs_pose2d_in_map()
            llc_ij = self.xy_to_ij(left_leg_pose2d_in_map_frame[:2])
            rlc_ij = self.xy_to_ij(right_leg_pose2d_in_map_frame[:2])
            # get circle 2d index of size r for each leg
            # TODO rotated stencil if ellipse
            circle_i, circle_j = self.circular_index_creator.make_circular_index(
                    agent.leg_radius, self.resolution_)
            lstencil_i = circle_i + llc_ij[0]
            lstencil_j = circle_j + llc_ij[1]
            rstencil_i = circle_i + rlc_ij[0]
            rstencil_j = circle_j + rlc_ij[1]
            stencil_i = np.hstack([lstencil_i, rstencil_i])
            stencil_j = np.hstack([lstencil_j, rstencil_j])
            # filter invalid indices
            mask = self.is_inside_ij(stencil_i, stencil_j)
            stencil_i = stencil_i[mask]
            stencil_j = stencil_j[mask]
            # paint stencil into map
            result_occupancy_grid[(stencil_i, stencil_j)] = 1
#                 from matplotlib import pyplot as plt
#                 plt.imshow(result_occupancy_grid)
#                 plt.plot(stencil_i[0], stencil_j[1], 'r-x')
#                 a_ij = self.xy_to_ij(m_a_T[:2])
#                 plt.plot(a_ij[0], a_ij[1], 'r-x')
#                 plt.show()
        return result_occupancy_grid


    def show(self):
        gridshow(self.occupancy_)


class LocalMap2D(Map2D):
    """ A fixed sized Map2D built from several observations,
    which remains centered around the current robot pose
    NOT THREADSAFE
    """

    def __init__(
        self, limits, resolution, sensor_model=None, max_observations=100
    ):  # limits, resolution in meters
        Map2D.__init__(self)
        self.limits = limits
        self.origin = limits[:, 0]
        self.resolution_ = resolution
        self.occupancy_ = 0.5 * np.ones(
            ((self.limits[:, 1] - self.limits[:, 0]) / self.resolution_).astype(int)
        )
        self.thresh_occupied_ = 0.9
        self.thresh_free = 0.1
        self.N_MAX_OBSERVATIONS = max_observations
        self.observations = [None] * self.N_MAX_OBSERVATIONS
        self.poses = [None] * self.N_MAX_OBSERVATIONS
        self.base_poses = [None] * self.N_MAX_OBSERVATIONS
        self.n_observations = 0
        if sensor_model is None:
            sensor_model = {"p_hit": 0.75, "p_miss": 0.25}
        self.sensor_model = sensor_model

    def ci_(self, i, n_observations=None, allow_overflow=False):
        """ converts from relative index i to circular index ci
        i = 0 refers to the latest observation
        i = -n refers to the nth latest observation
        """
        if n_observations is None:
            n_observations = self.n_observations
        if i > 0:
            raise IndexError
        if i <= -self.N_MAX_OBSERVATIONS and not allow_overflow:
            raise IndexError
        if i <= -n_observations and not allow_overflow:
            raise IndexError
        return (n_observations - 1 + i) % self.N_MAX_OBSERVATIONS

    def all_cis(self):
        """ returns a list of all available observation circular indices,
        from oldest to newest """
        return [
            self.ci_(i)
            for i in range(1 - min(self.n_observations, self.N_MAX_OBSERVATIONS), 1)
        ]

    def add_observation(self, scan, pose, base_pose=None):
        """ Adds a scan to the current observation list """
        self.n_observations += 1
        self.observations[self.ci_(0)] = scan
        self.poses[self.ci_(0)] = pose
        self.base_poses[self.ci_(0)] = base_pose
        return

    def generate(self):
        """ Generates latest occupancy and cross_entropy_error from observations """
        if self.n_observations == 0:
            return None
        current_pose = self.base_poses[self.ci_(0)]
        # Combine observations
        inc_occupancy = self.occupancy_ * 1.
        errors = []
        r = np.array(
            [  # rotation for oTa, o = odom frame, a = current_scan frame
                [np.cos(current_pose[2]), np.sin(current_pose[2])],
                [-np.sin(current_pose[2]), np.cos(current_pose[2])],
            ]
        )
        for ci in self.all_cis():
            # get relative pose
            rel_pose = self.poses[ci] - current_pose
            rel_pose[:2] = np.sum(
                r * rel_pose[:2], axis=-1
            )  # rel_pose x y in current_scan frame
            # reverse raytrace scan
            scan = self.observations[ci]
            hits_and_misses = self.occupancy_ * 1.
            self.reverse_raytrace_scan(scan, rel_pose, hits_and_misses)
            # reinforce latest scan?
            if ci == self.ci_(0):
                phit = 0.999
            else:
                phit = self.sensor_model["p_hit"]
            # merge scan with history
            compiled_insert_scan(
                hits_and_misses,
                inc_occupancy,
                phit=phit,
                pmiss=self.sensor_model["p_miss"],
            )
        # fill variables
        result = Map2D()
        result.occupancy_ = inc_occupancy
        result.resolution_ = self.resolution_
        result.origin = self.origin
        result.thresh_occupied_ = self.thresh_occupied_
        result.thresh_free = self.thresh_free
        return result

    def reverse_raytrace_scan(self, scan, pose, occupancy):
        """ Generates the occupancy field from the laser data.
        this returns a map with the same pose and dimensions as this one,
        however the occupancy is based on the hits and misses of the laser scan
        the occupancy values are 0 for misses, 1 for hits.
        pose: scan pose in map frame
        """
        ij_hits, ij_laser_orig = self.scan_in_ij_frame(
            scan, pose, handle_no_return_as_far=True
        )
        compiled_reverse_raytrace(ij_hits, ij_laser_orig, occupancy)
        return occupancy

    def scan_in_ij_frame(self, scan, pose, handle_no_return_as_far=False):
        angles = (
                np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)[:len(scan.ranges)] 
                + pose[2]
        )
        if handle_no_return_as_far:
            ranges = [
                1000. if r == 0 else r for r in scan.ranges
            ]  # find far away points
        else:
            ranges = scan.ranges
        xy_hits = (
            np.array(ranges) * np.array([np.cos(angles), np.sin(angles)])
        ).T + pose[:2]
        ij_hits = self.xy_to_ij(xy_hits, clip_if_outside=False)
        ij_laser_orig = (
            (-self.origin / self.resolution_)  # ij_coordinates of point where x, y = 0
            + pose[:2]
        ).astype(int)
        return ij_hits, ij_laser_orig

    def insert_scan(self, hits_and_misses, prior):
        """ Instead of moving the scan to the map pose,
        the previous map gets moved to the new scan pose"""
        phit = self.sensor_model["p_hit"]
        pmiss = self.sensor_model["p_miss"]
        hits_and_misses
        p_measured = hits_and_misses[:]
        p_measured[
            p_measured == 0
        ] = (
            pmiss
        )  # TODO: could make sense to update pmiss in raytrace: having lower prob where several beams pass
        p_measured[p_measured == 1] = phit
        posterior = (
            p_measured * prior / (p_measured * prior + (1 - p_measured) * (1 - prior))
        )
        return posterior

    def cross_entropy_error(self, scan, pose, prior):
        ij_hits, ij_laser_orig = self.scan_in_ij_frame(scan, pose)
        valid_mask = np.all(
            np.logical_and(ij_hits >= 0, ij_hits < self.occupancy_.shape), axis=-1
        )
        ij_hits_in_map = ij_hits[valid_mask]
        priors = prior[tuple(ij_hits_in_map.T)]
        pz = self.sensor_model["p_hit"]
        error = -priors * np.log(pz) - (1 - priors) * np.log(1 - pz)
        return error, ij_hits_in_map

    def copy(self):
        """
        Meant to be used with, for example, the case where the localmap has
        observations added from one thread at high rate, and must have .generate() called
        from another thread. This copy will efficiently ensure the observations are frozen.
        BEWARE, changing the observations themselves will also affect the original! Shallow copy
        """
        # deepcopy everything except observations
        copy = LocalMap2D(self.limits, self.resolution_)
        copy.limits = self.limits * 1.  # deep copy
        copy.origin = self.origin * 1.
        copy.resolution_ = self.resolution_
        copy.thresh_occupied_ = self.thresh_occupied_
        copy.thresh_free = self.thresh_free
        copy.N_MAX_OBSERVATIONS = self.N_MAX_OBSERVATIONS
        copy.observations = self.observations[:]  # BEWARE: shallow copy for efficiency
        copy.poses = self.poses[:]
        copy.base_poses = self.base_poses[:]
        copy.n_observations = self.n_observations
        copy.sensor_model = self.sensor_model.copy()  # deep copy
        copy.occupancy_ = self.occupancy_ * 1.
        return copy


def gridshow(*args, **kwargs):
    """ utility function for showing 2d grids in matplotlib,
    wrapper around pyplot.imshow

    use 'extent' = [-1, 1, -1, 1] to change the axis values """
    from matplotlib import pyplot as plt
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'
    if not 'cmap' in kwargs:
        kwargs['cmap'] = plt.cm.Greys
    return plt.imshow(*(arg.T if i == 0 else arg for i, arg in enumerate(args)), **kwargs)

class SimAgent(object):
    def __init__(self):
        self.pose_2d_in_map_frame = np.array([0, 0, 0])
        self.type = "legs"
        self.state = None
        self.leg_radius = 0.03 # [m]

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
            left_leg_pose2d_in_map_frame = pose2d.apply_tf_to_pose(
                    left_leg_pose2d_in_agent_frame, m_a_T)
            right_leg_pose2d_in_map_frame = pose2d.apply_tf_to_pose(
                    right_leg_pose2d_in_agent_frame, m_a_T)
            return left_leg_pose2d_in_map_frame, right_leg_pose2d_in_map_frame
        else:
            raise NotImplementedError

if __name__ == "__main__":
    import doctest

    doctest.testmod()


if __name__ == "test render agent":
    import numpy as np
    from matplotlib import pyplot as plt

    plt.ion()
    plt.figure()
    for i in range(100):
        plt.cla()
        plt.scatter([0], [0])

        r0 = 1.5
        lmbda = np.random.random() * np.pi * 2
        r0sq = r0**2
        R = np.random.random()
        phi = np.linspace(0, 2*np.pi, 1000)
        first_term = r0 * np.cos(phi - lmbda)
        sqrt_inner = r0sq * np.cos(phi - lmbda)**2 - r0sq + R**2
        sqrt_inner[sqrt_inner < 0] = np.inf
        radii_solutions_a = first_term - np.sqrt(sqrt_inner)
        radii_solutions_b = first_term + np.sqrt(sqrt_inner)
        radii_solutions_a[radii_solutions_a < 0] = np.inf
        radii_solutions_b[radii_solutions_b < 0] = np.inf
        r = np.minimum(radii_solutions_a, radii_solutions_b)
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        plt.plot(x, y)

        phiext = [
                lmbda - np.arccos( np.sqrt(r0sq - R**2) / r0 ),
                lmbda + np.arccos( np.sqrt(r0sq - R**2) / r0 ),
        #         lmbda + np.arccos( - np.sqrt(r0sq - R**2) / r0 ),
        #         lmbda - np.arccos( - np.sqrt(r0sq - R**2) / r0 ),
                ]

        colors = ['r', 'g']
        for ph, c in zip(phiext, colors):
            x = [0, r0 * np.cos(ph)]
            y = [0, r0 * np.sin(ph)]
            plt.plot(x, y, '-'+c)


        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.show()

        plt.pause(0.1)
