import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import copy

source = o3d.io.read_point_cloud("/home/ohheemin/Downloads/source.ply")
target = o3d.io.read_point_cloud("/home/ohheemin/Downloads/target.ply")

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
                                     
                                      
threshold = 1.0
trans_init = np.identity(4)
draw_registration_result(source, target, trans_init)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)

# Initial alignment
# RegistrationResult with fitness=1.046892e-01, inlier_rmse=8.311949e-01, and correspondence_set size of 192
# Access transformation to get result.


########################## point-to-point ICP ##############################
print("\n\nApply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

# Apply point-to-point ICP
# RegistrationResult with fitness=9.411123e-01, inlier_rmse=2.790346e-01, and correspondence_set size of 1726
# Access transformation to get result.
# Transformation is:
# [[ 0.98046308  0.07335189 -0.18251481  1.32404862]
#  [-0.07317525  0.99728929  0.00771128 -0.11099725]
#  [ 0.1825857   0.00579494  0.98317286 -1.42716579]
#  [ 0.          0.          0.          1.        ]]

############################################################################


########################## point-to-plane ICP ##############################
print("\n\nApply point-to-plane ICP")
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)

# Apply point-to-plane ICP
# RegistrationResult with fitness=9.323882e-01, inlier_rmse=2.793996e-01, and correspondence_set size of 1710
# Access transformation to get result.
# Transformation is:
# [[ 9.66789671e-01  1.58338593e-01 -2.00615607e-01  1.36196330e+00]
#  [-1.54972254e-01  9.87384826e-01  3.24777823e-02 -2.02984836e-01]
#  [ 2.03227292e-01 -3.09331810e-04  9.79131540e-01 -1.36642907e+00]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
############################################################################