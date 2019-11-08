from distutils.core import setup
from Cython.Build import cythonize
import os


setup(
    name="pymap2d",
    description='Tools for 2D maps',
    author='Daniel Dugas',
    version='1.0',
    py_modules=['map2d', 'pose2d', 'circular_index', 'map2d_ros_tools'],
    ext_modules = cythonize("CMap2D.pyx", annotate=True),
)
