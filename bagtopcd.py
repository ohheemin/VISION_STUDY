import os
import struct
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2, PointField
import pcl

def rosbag_to_pcd(bag_folder_path, topic_name, output_pcd_file):
    """
    Convert a ROS 2 bag folder to a PCD file.
    """
    if not os.path.exists(bag_folder_path):
        raise FileNotFoundError(f"Bag folder not found: {bag_folder_path}")

    # Initialize the reader for the ROS 2 bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_folder_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
    reader.open(storage_options, converter_options)

    # Get metadata about the bag file and print all topics to check
    topic_types = reader.get_all_topics_and_types()
    topic_type_dict = {topic.name: topic.type for topic in topic_types}
    
    print(f"Available topics: {list(topic_type_dict.keys())}")
    
    if topic_name not in topic_type_dict:
        raise ValueError(f"Topic '{topic_name}' not found in the bag folder.")
    
    print(f"Extracting messages from topic: {topic_name}")

    # Extract LiDAR points (x, y, z) from the bag file
    points = []
    while reader.has_next():
        (topic, msg, t) = reader.read_next()
        if topic == topic_name:
            # Parse PointCloud2 message
            if isinstance(msg, PointCloud2):
                points.extend(parse_pointcloud2(msg))
    
    if len(points) == 0:
        print(f"No data found in topic {topic_name}")
        return

    # Save data to PCD file
    save_points_to_pcd(points, output_pcd_file)
    print(f"PCD file saved at: {output_pcd_file}")


def parse_pointcloud2(msg):
    """
    Parse a PointCloud2 message into a list of (x, y, z) points.
    
    :param msg: PointCloud2 message.
    :return: List of (x, y, z) tuples.
    """
    # Get the point cloud data
    points = []
    point_step = msg.point_step
    fields = {field.name: field.offset for field in msg.fields}

    # Check that we have x, y, z data
    if 'x' in fields and 'y' in fields and 'z' in fields:
        for i in range(msg.width):
            offset = i * point_step
            x = struct.unpack_from('f', msg.data, offset + fields['x'])[0]
            y = struct.unpack_from('f', msg.data, offset + fields['y'])[0]
            z = struct.unpack_from('f', msg.data, offset + fields['z'])[0]
            points.append((x, y, z))
    return points

def save_points_to_pcd(points, output_pcd_file):
    """
    Save points to a PCD file.
    
    :param points: List of (x, y, z) points.
    :param output_pcd_file: Path to save the output PCD file.
    """
    # Convert to PCL format
    cloud = pcl.PointCloud()
    cloud.from_list(points)

    # Save to PCD file
    cloud.to_file(output_pcd_file)

def main():
    bag_folder_path = "/home/ohheemin/ros2_ws/lidar_lane_no_noise"  # Input bag folder path
    topic = "/points"  # Topic name for LiDAR data
    output_pcd = "vision_task.pcd"  # Output PCD file path

    rosbag_to_pcd(bag_folder_path, topic, output_pcd)

if __name__ == '__main__':
    main()
