from __future__ import print_function
from builtins import input
import CMap2D
import numpy as np
import matplotlib.pyplot as plt
from map2d import gridshow
from pyniel.testing_tools.testgen import freeze_test_case, load_test_case

# test case 1
office2d = CMap2D.CMap2D(".", "office_full")
args = (office2d,)
def coarse_x3_occupancy(cmap2d):
    return cmap2d.as_coarse_map2d().as_coarse_map2d().as_coarse_map2d().occupancy()
coarse = coarse_x3_occupancy(*args)
plt.figure()
gridshow(coarse)
plt.pause(0.1)
print("Freeze test results [y/N]?")
keys = input(">> ")
if keys == 'y':
    freeze_test_case(coarse_x3_occupancy, args, folder="./test_data", case_name="office_full_coarse", omit_input=True)
else:
    pass
plt.close('all')

# test case 1
coarse = office2d.as_coarse_map2d(n=3)
grid = coarse.as_sdf()
plt.figure()
gridshow(grid)
plt.pause(0.1)
print("Freeze test results [y/N]?")
keys = input(">> ")
if keys == 'y':
    freeze_test_case(coarse.as_sdf, (), folder="./test_data", case_name="office_full_coarse_as_sdf", omit_input=True)
else:
    pass
plt.close('all')

# test case 2
default = CMap2D.CMap2D(".", "tests")
ij = np.array([[0, 0]], dtype=np.int64)
args = (ij[0],)
kwargs = {'inv_value':-1, 'connectedness': 16}
grid16 = default.dijkstra(*args, **kwargs)
gridshow(np.mod(np.maximum(grid16,0),2))
plt.pause(0.1)
print("Freeze test results [y/N]?")
keys = input(">> ")
if keys == 'y':
    freeze_test_case(default.dijkstra, args, kwargs=kwargs, folder="./test_data", case_name="default_dijkstra_16")
else:
    pass
plt.close('all')

# test case 3
default = CMap2D.CMap2D(".", "tests")
ij = np.array([[0, 0]], dtype=np.int64)
args = (ij[0],)
kwargs = {'inv_value':-1, 'connectedness': 32}
grid32 = default.dijkstra(*args, **kwargs)
gridshow(np.mod(np.maximum(grid32,0),2))
plt.pause(0.1)
print("Freeze test results [y/N]?")
keys = input(">> ")
if keys == 'y':
    freeze_test_case(default.dijkstra, args, kwargs=kwargs, folder="./test_data", case_name="default_dijkstra_32")
else:
    pass
plt.close('all')
