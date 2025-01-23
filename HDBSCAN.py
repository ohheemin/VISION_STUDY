import matplotlib.pyplot as plt
import time
import numpy as np
import open3d as o3d
import hdbscan

pcd_path = "/home/ohheemin/Downloads/000000.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
plt.ion()

# CLUSTERING WITH DBSCAN
t3 = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
clusterer.fit(np.array(pcd.points))
labels = clusterer.labels_

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using HDBSCAN {t4 - t3}')
o3d.visualization.draw_geometries([pcd])
