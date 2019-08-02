# pymap2d

Map2D is a simple python 2D gridmap class,
able to load 2d maps from ROS .yaml/.pgm maps.

Map2D provides:
- simple xy <-> ij coordinate conversions,
- implementation of the dijkstra algorithm.
- 2D ESDF calculation

CMap2D is a Cython-based performance optimized version of Map2D,
providing fast 2D ESDF calculation from occupancy grids.

## Dependency: Cython
```
$ pip install Cython
```

## Installation:
Inside this project root folder:
```
$ pip install .
```
