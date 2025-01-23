import open3d as o3d
import numpy as np
import time

npy_file = "/home/ohheemin/macaron_06/vision_task.npy"
pcd_data = np.load(npy_file)

# 데이터 차원 확인 및 변환
if pcd_data.ndim == 1:
    if len(pcd_data) % 3 == 0:
        pcd_data = pcd_data.reshape(-1, 3)
    else:
        raise ValueError("The data cannot be reshaped into a 3D point cloud format. Check the input file.")
elif pcd_data.ndim == 2 and pcd_data.shape[1] >= 3:
    pcd_data = pcd_data[:, :3]  # 좌표 정보만 사용
else:
    raise ValueError("Unexpected data format. Ensure the input file contains 3D point cloud data.")

# Open3D로 PointCloud 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
if pcd_data.shape[1] >= 6:  # 색상 정보가 있는 경우
    pcd.colors = o3d.utility.Vector3dVector(pcd_data[:, 3:6])

# Downsampling
pcd_1 = pcd.voxel_down_sample(voxel_size=0.05)

# Remove outliers
pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=20, radius=0.3)

# Segment plane with RANSAC
plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

# Clustering with HDBSCAN
import matplotlib.pyplot as plt
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
clusterer.fit(np.array(pcd_3.points))
labels = clusterer.labels_

max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Generate 3D Bounding Box
import pandas as pd
bbox_objects = []
indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

MAX_POINTS = 300
MIN_POINTS = 50

for i in range(0, len(indexes)):
    nb_points = len(pcd_3.select_by_index(indexes[i]).points)
    if MIN_POINTS < nb_points < MAX_POINTS:
        sub_cloud = pcd_3.select_by_index(indexes[i])
        bbox_object = sub_cloud.get_axis_aligned_bounding_box()
        bbox_object.color = (0, 0, 1)
        bbox_objects.append(bbox_object)

print("Number of Bounding Boxes: ", len(bbox_objects))

list_of_visuals = [pcd_3] + bbox_objects
o3d.visualization.draw_geometries(list_of_visuals)
