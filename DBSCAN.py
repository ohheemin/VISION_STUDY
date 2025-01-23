import matplotlib.pyplot as plt
import time
import numpy as np
import open3d as o3d

pcd_path = "/home/ohheemin/Downloads/000000.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
t3 = time.time()
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.60, min_points=30, print_progress=False))

max_label = labels.max()
print(f'point cloud has {max_label + 1} clusters')
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
t4 = time.time()
print(f'Time to cluster outliers using DBSCAN {t4 - t3}')
o3d.visualization.draw_geometries([pcd])