import numpy
from setuptools import setup
from Cython.Build import cythonize

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="pymap2d",
    description='Tools for 2D maps',
    author='Daniel Dugas',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/danieldugas/pymap2d",
    version='0.1.1',
    py_modules=['map2d', 'pose2d', 'circular_index', 'map2d_ros_tools'],
    ext_modules=cythonize("CMap2D.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)
