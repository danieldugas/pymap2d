# pymap2d

pymap2d is a Cython-based fast toolbox for 2d grid maps.

The CMap2D class provides:
- simple xy <-> ij coordinate conversions
- implementation of the dijkstra / fastmarch algorithm
- fast 2D distance transform (ESDF)
- conversions:
  - to/from polygon vertices
  - from ROS occupancy map or lidar scan message
  - serialization to/from dict

![pymap2d][media/pymap2d.png]

Note: rather than carefully designed, this codebase was chaotically grown. 
It is in dire need of refactoring / documentation. I hope it still proves useful.

## Dependency: Cython
```
$ pip install numpy Cython
```

## Installation:
Inside this project root folder:
```
$ pip install pymap2d
```
