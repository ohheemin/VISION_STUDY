import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# LiDAR 데이터 로드
lidar_data = np.load("/home/ohheemin/macaron_06/lidar_all_scans_noise.npy", allow_pickle=True)

def extract_two_lane_lines(lidar_data, intensity_threshold=60, roi_x=(-10, 10), roi_y=(2, 50)):
    for i, pt_cloud in enumerate(lidar_data):
        if pt_cloud.dtype.names == ('x', 'y', 'z', 'intensity'):
            # 강도 필터링 (intensity > threshold)
            filtered_points = pt_cloud[pt_cloud['intensity'] > intensity_threshold]

            # ROI (차선이 존재할 가능성이 높은 영역)
            x, y = filtered_points['x'], filtered_points['y']
            roi_mask = (roi_x[0] < x) & (x < roi_x[1]) & (roi_y[0] < y) & (y < roi_y[1])
            x_roi, y_roi = x[roi_mask], y[roi_mask]

            if x_roi.size > 0:
                plt.figure(figsize=(10, 6))
                plt.scatter(x_roi, y_roi, c=filtered_points['intensity'][roi_mask], cmap='viridis', s=10, label="Lane Points")

                # 첫 번째 직선 검출 (RANSAC)
                model1 = RANSACRegressor()
                X1 = x_roi.reshape(-1, 1)
                model1.fit(X1, y_roi)
                y_pred1 = model1.predict(X1)
                plt.plot(x_roi, y_pred1, color='red', linewidth=2, label="Lane Line 1")

                # 첫 직선 점들 제거, 두 번째 직선 검출
                inlier_mask = model1.inlier_mask_
                x_roi_remain, y_roi_remain = x_roi[~inlier_mask], y_roi[~inlier_mask]

                if x_roi_remain.size > 0:
                    model2 = RANSACRegressor()
                    X2 = x_roi_remain.reshape(-1, 1)
                    model2.fit(X2, y_roi_remain)
                    y_pred2 = model2.predict(X2)
                    plt.plot(x_roi_remain, y_pred2, color='blue', linewidth=2, label="Lane Line 2")

                plt.title(f"Two lane version - {i}")
                plt.xlabel("X (meters)")
                plt.ylabel("Y (meters)")
                plt.colorbar(label="Intensity")
                plt.legend()
                plt.grid(True)
                plt.show()

if lidar_data is not None:
    extract_two_lane_lines(lidar_data)
else:
    print("Error: Failed to load LiDAR data.")
