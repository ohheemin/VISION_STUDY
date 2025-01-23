import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
import hdbscan
import pandas as pd

def process_point_cloud(pcd_data):
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

    # 색상 정보가 있는 경우, 색상 추가 (컬러 데이터가 있을 경우 4번째, 5번째, 6번째 컬럼에)
    if pcd_data.shape[1] >= 6:  
        pcd.colors = o3d.utility.Vector3dVector(pcd_data[:, 3:6])

    # Downsampling with a smaller voxel size
    pcd_1 = pcd.voxel_down_sample(voxel_size=0.1)  # Smaller voxel size
    o3d.visualization.draw_geometries([pcd_1])  # Visualize after downsampling

    # Remove outliers with looser parameters
    pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=5, radius=1.5)  # Loosened outlier removal
    o3d.visualization.draw_geometries([pcd_2])  # Visualize after outlier removal

    # Check if there are enough points before proceeding to plane segmentation or clustering
    if len(pcd_2.points) == 0:
        print("No points remaining after downsampling and outlier removal. Skipping further processing.")
        return

    # Skip plane segmentation if too few points
    if len(pcd_2.points) < 3:
        print("Not enough points after downsampling and outlier removal, skipping plane segmentation.")
        pcd_3 = pcd_2  # Use whatever points remain
    else:
        # Segment plane with RANSAC
        plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
        pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

    o3d.visualization.draw_geometries([pcd_3])  # Visualize after plane segmentation

    # Clustering with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
    clusterer.fit(np.array(pcd_3.points))
    labels = clusterer.labels_

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
    colors[labels < 0] = 0
    pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Check for empty point cloud before bounding box generation
    if len(pcd_3.points) == 0:
        print("No points left after clustering. Skipping bounding box generation.")
        return

    # Generate 3D Bounding Box
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

# 예시: numpy 배열로 된 포인트 클라우드 데이터 로드
npy_file = "/home/ohheemin/macaron_06/lidar_all_scans.npy"
pcd_data = np.load(npy_file)

# 포인트 클라우드 처리 함수 호출
process_point_cloud(pcd_data)
