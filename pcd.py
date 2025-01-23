import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the point cloud using Open3D
pcd = o3d.io.read_point_cloud("/home/ohheemin/Downloads/000000.pcd")
pcd_np = np.asarray(pcd.points)

# Create a 3D scatter plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z coordinates
x = pcd_np[:, 0]
y = pcd_np[:, 1]
z = pcd_np[:, 2]

# Plot the points
ax.scatter(x, y, z, s=1, c=z, cmap='viridis', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
