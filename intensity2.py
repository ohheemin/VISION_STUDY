import numpy as np
import matplotlib.pyplot as plt

# Load the LiDAR data
lidar_data = np.load("/home/ohheemin/macaron_06/lidar_all_scans_no_noise.npy", allow_pickle=True)

# Filter and visualize the points where the intensity is greater than 80
def filter_and_visualize_data(lidar_data):
    for i, pt_cloud in enumerate(lidar_data):
        # Check if the point cloud has the expected structure
        if pt_cloud.dtype.names == ('x', 'y', 'z', 'intensity'):
            # Filter points where the intensity is greater than 80
            filtered_points = pt_cloud[pt_cloud['intensity'] > 80]

            if filtered_points.size > 0:
                # Visualize the filtered points
                plt.figure(figsize=(10, 6))
                plt.scatter(filtered_points['x'], filtered_points['y'], c=filtered_points['intensity'], cmap='viridis', s=10)
                plt.title(f"Filtered Points (Intensity > 70) - Frame {i}")
                plt.xlabel("X (meters)")
                plt.ylabel("Y (meters)")
                plt.colorbar(label="Intensity")
                plt.grid(True)
                plt.show()

if lidar_data is not None:
    filter_and_visualize_data(lidar_data)
else:
    print("Error: Failed to load LiDAR data.")
