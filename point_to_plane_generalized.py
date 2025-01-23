import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import copy

demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])                                 

print("\n\nApply point-to-plane ICP")
demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
threshold =0.02
trans_init = np.identity(4)

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
# RegistrationResult with fitness=1.000000e+00, inlier_rmse=2.660825e-01, and correspondence_set size of 198835
# Access transformation to get result.
# Transformation is:
# [[ 0.80415634  0.0446264  -0.5927403   1.71318193]
#  [-0.28280136  0.90581483 -0.31547248  1.26215451]
#  [ 0.52283455  0.42131696  0.74103714 -1.38225719]
#  [ 0.          0.          0.          1.        ]]

print("\n\nApply Generalized ICP")
demo_icp_pcds = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

trans_init = np.identity(4)
source_down_sample = source.voxel_down_sample(0.1)
target_down_sample = target.voxel_down_sample(0.1)

max_correspondence_distances = 0.3
reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
    source, target, max_correspondence_distances, trans_init, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

print(reg_gicp)
print("Transformation is:")
print(reg_gicp.transformation)
draw_registration_result(source, target, reg_gicp.transformation)

# Apply Generalized ICP
# RegistrationResult with fitness=7.213871e-01, inlier_rmse=5.392053e-02, and correspondence_set size of 143437
# Access transformation to get result.
# Transformation is:
# [[ 0.84047262  0.00661921 -0.54181359  0.64381443]
#  [-0.14759452  0.96491234 -0.2171636   0.81044623]
#  [ 0.52136516  0.26248878  0.81195936 -1.48451717]
#  [ 0.          0.          0.          1.        ]]