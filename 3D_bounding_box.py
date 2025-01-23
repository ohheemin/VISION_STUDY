import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# 포인트 클라우드 파일 경로
pcd_path = "/home/ohheemin/ws/output.pcd"

pcd = o3d.io.read_point_cloud(pcd_path)

#다운샘플링
pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.03)
#아웃라이더 제거 과정
pcd_nooutliers, inliers = pcd_downsampled.remove_radius_outlier(nb_points=20, radius=1)

#세그먼트로 란삭
plane_model, road_inliers = pcd_nooutliers.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
pcd_ransac =pcd_nooutliers.select_by_index(road_inliers, invert=True)

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, gen_min_span_tree=True)
clusterer.fit(np.array(pcd_ransac.points))
labels = clusterer.labels_
max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
pcd_ransac.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 바운딩 박스를 저장할 리스트
bbox_objects = []
# 클러스터별 인덱스 계산
indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()                                    

# 클러스터 조건
MAX_POINTS = 300
MIN_POINTS = 50

# 클러스터별 바운딩 박스 생성
for i in range(0,len(indexes)):
    nb_points = len(pcd_ransac.select_by_index(indexes[i]).points)
    if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
        sub_cloud = pcd_ransac.select_by_index(indexes[i])
        bbox_object = sub_cloud.get_axis_aligned_bounding_box()
        bbox_object.color = (0, 0, 1)  # 바운딩 박스 색상: 파란색
        bbox_objects.append(bbox_object)

# 결과 출력
print("Number of Bounding Boxes:", len(bbox_objects))

# 시각화 리스트 생성
list_of_visuals = []
list_of_visuals.append(pcd_ransac)
list_of_visuals.extend(bbox_objects)

# 시각화
o3d.visualization.draw_geometries(list_of_visuals)



