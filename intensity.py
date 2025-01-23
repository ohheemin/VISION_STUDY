import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

"""lidar data load
"""
def load_lidar_data(file_path):
    data = np.load(file_path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        processed_data = []
        for i, frame in enumerate(data):
            if frame.ndim == 1:  # 1D 배열
                if frame.size % 4 != 0:
                    print(f"Error: Frame {i} cannot be reshaped to (N, 4). Shape: {frame.shape}")
                    continue  # 잘못된 데이터일 때
                try:
                    frame = frame.reshape(-1, 4)  # (N, 4)로 변환
                except ValueError as e:
                    print(f"Error reshaping frame {i}: {e}")
                    continue
            elif frame.ndim != 2 or frame.shape[1] != 4:
                print(f"Error: Frame {i} has invalid shape: {frame.shape}")
                continue

            processed_data.append(frame)
        return processed_data
    return None

"""ROI 추출 및 지면 제거
"""
def preprocess_point_cloud(pt_cloud, xlim, ylim, zlim):
    """포인트 클라우드가 어레이일 때
    """
    # Check if it's a structured numpy array with named fields
    if isinstance(pt_cloud, np.ndarray) and pt_cloud.dtype.names:
        pt_cloud = np.column_stack([pt_cloud['x'], pt_cloud['y'], pt_cloud['z'], pt_cloud['intensity']]).astype(np.float32)
    elif pt_cloud.shape[1] == 12:  # Handling the case with 12 columns
        pt_cloud = pt_cloud[:, :4].astype(np.float32)  # Extract x, y, z, intensity columns
    else:
        raise ValueError("Unsupported point cloud format")

    """x,y,z가 (N,4)포맷인 것을 확인
    """
    if pt_cloud.ndim != 2 or pt_cloud.shape[1] != 4:
        raise ValueError(f"Invalid point cloud format. Shape: {pt_cloud.shape}")

    """ROI 선택
    """
    roi_mask = np.logical_and(
        np.logical_and(np.greater_equal(pt_cloud[:, 0], xlim[0]), np.less_equal(pt_cloud[:, 0], xlim[1])),
        np.logical_and(np.logical_and(np.greater_equal(pt_cloud[:, 1], ylim[0]), np.less_equal(pt_cloud[:, 1], ylim[1])),
                       np.logical_and(np.greater_equal(pt_cloud[:, 2], zlim[0]), np.less_equal(pt_cloud[:, 2], zlim[1])))
    )
    cropped_pc = pt_cloud[roi_mask]

    """평면 맞춤
    """
    def plane_fit(points):
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        b = points[:, 2]
        solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return solution

    plane_params = plane_fit(cropped_pc[:, :3])  # Fit on x, y, z only
    ground_mask = np.abs(cropped_pc[:, 2] - (
        plane_params[0] * cropped_pc[:, 0] +
        plane_params[1] * cropped_pc[:, 1] +
        plane_params[2])) < 0.1

    """비지면 점과 지면 점을 반환
    """
    return cropped_pc[~ground_mask], cropped_pc[ground_mask]


def compute_histogram(point_cloud, bin_resolution=0.2):
    y_limits = [np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])]
    num_bins = int((y_limits[1] - y_limits[0]) / bin_resolution)
    hist_vals = np.zeros(num_bins)
    bin_centers = np.linspace(y_limits[0], y_limits[1], num_bins)

    for i, y_val in enumerate(bin_centers):
        bin_mask = (point_cloud[:, 1] > y_val - bin_resolution / 2) & (point_cloud[:, 1] <= y_val + bin_resolution / 2)
        hist_vals[i] = np.sum(point_cloud[bin_mask][:, 3])  # intensity (column 3 now)

    return hist_vals, bin_centers

"""시각화 함수
"""
def visualize_results(non_ground_points, left_lane_pts, right_lane_pts):
    if non_ground_points.ndim != 2 or left_lane_pts.ndim != 2 or right_lane_pts.ndim != 2:
        print("Skipping visualization due to invalid input shapes.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(non_ground_points[:, 0], non_ground_points[:, 1], s=1, c='gray', label='All Points', alpha=0.5)
    plt.scatter(left_lane_pts[:, 0], left_lane_pts[:, 1], s=5, c='blue', label='Left Lane Points')
    plt.scatter(right_lane_pts[:, 0], right_lane_pts[:, 1], s=5, c='red', label='Right Lane Points')
    plt.title("Lane Detection Results")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid()
    plt.show()

"""라이다 데이터 처리
"""
def process_lidar_data(lidar_data):
    for i, pt_cloud in enumerate(lidar_data):
        """1D 배열인 경우
        """
        if pt_cloud.ndim == 1:  
            if pt_cloud.size % 4 != 0:
                print(f"Error: Frame {i} cannot be reshaped to (N, 4). Shape: {pt_cloud.shape}")
                continue  
            try:
                pt_cloud = pt_cloud.reshape(-1, 4)  # (N, 4)로 변환
            except ValueError as e:
                print(f"Error reshaping frame {i}: {e}")
                continue

        """효용성이 없는 데이터는 컷
        """
        if pt_cloud.ndim != 2 or pt_cloud.shape[1] != 4:
            print(f"Error: Invalid shape for frame {i}: {pt_cloud.shape}")
            continue

        print(f"Processing frame {i}: shape {pt_cloud.shape}")
        non_ground_points, _ = preprocess_point_cloud(pt_cloud, [5, 40], [-3, 3], [-4, 1])

        """히스토그램 계산
        """
        hist_vals, bin_centers = compute_histogram(non_ground_points)
        peaks = find_peaks(hist_vals)[0]

        if len(peaks) < 2:
            print("Not enough peaks detected for lane fitting.")
            continue  

        left_lane_pts = non_ground_points[non_ground_points[:, 1] > bin_centers[peaks[0]]]
        right_lane_pts = non_ground_points[non_ground_points[:, 1] < bin_centers[peaks[1]]]

        if len(left_lane_pts) < 2 or len(right_lane_pts) < 2:
            print("Not enough points for polynomial fitting.")
            continue 

        print(f"Frame {i}: Left lane points: {len(left_lane_pts)}, Right lane points: {len(right_lane_pts)}")

        visualize_results(non_ground_points, left_lane_pts, right_lane_pts)

    print("All frames processed.")

"""npy 파일을 로드함
"""
if __name__ == "__main__":
    lidar_data = np.load("/home/ohheemin/macaron_06/lidar_all_scans.npy", allow_pickle=True)

    if lidar_data is not None:
        process_lidar_data(lidar_data)  
    else:
        print("Error: Failed to load LiDAR data.")
