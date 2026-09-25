import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

#import pcl
import sys

def pointcloud2_to_xyz_array(cloud_msg):
    """Convert PointCloud2 message to numpy array of XYZ coordinates."""
    return point_cloud2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)

def xyz_array_to_pointcloud2(points, frame_id="ves/left/base"):
    """Convert numpy array of XYZ coordinates to PointCloud2 message."""
    header = Header()
    header.frame_id = frame_id
    
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    
    return point_cloud2.create_cloud(header, fields, points)



def create_rotation_matrix(angle, axis):
    """Create a 3D rotation matrix for rotation about specified axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ], dtype=np.float32)  # Ensure float32
    elif axis.lower() == 'y':
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float32)  # Ensure float32
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], dtype=np.float32)  # Ensure float32
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def shift_pointcloud(input_pc, x_offset=0.0, y_offset=0.0, z_offset=0.0, rotation_angle=0.0, rotation_axis='z'):
    # Ensure input_pc is a numpy array
    if not isinstance(input_pc, np.ndarray):
        raise TypeError("Expected input_pc to be a numpy.ndarray")

    # Apply translation
    translation_matrix = np.array([x_offset, y_offset, z_offset])
    points = input_pc.astype(np.float32) + translation_matrix

    # Apply rotation if needed
    if rotation_angle != 0.0:
        # Define rotation matrix based on the axis
        if rotation_axis == 'z':
            rotation_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])
        elif rotation_axis == 'y':
            rotation_matrix = np.array([
                [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                [0, 1, 0],
                [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
            ])
        elif rotation_axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)]
            ])
        else:
            raise ValueError("Invalid rotation axis. Choose from 'x', 'y', or 'z'.")

        # Apply rotation
        points = np.dot(points, rotation_matrix.T)

    return points


def lop_pc(point_cloud):
    """
    Removes all points in the point cloud with a z value greater than the mean z value.
    Parameters:
    - point_cloud (np.ndarray): Nx3 array where each row is a point [x, y, z]
    Returns:
    - filtered_point_cloud (np.ndarray): Filtered point cloud with z <= mean(z)
    """
    z_mean = np.mean(point_cloud[:, 2])
    mask = point_cloud[:, 2] <= z_mean - 0.0015
    filtered_point_cloud = point_cloud[mask]
    return filtered_point_cloud

# if __name__ == "__main__":
#     if len(sys.argv) < 5:
#         print("Usage: python3 shift_pointcloud.py input_file output_file x_offset y_offset z_offset [rotation_angle rotation_axis]")
#         print("rotation_angle should be in radians")
#         print("rotation_axis should be 'x', 'y', or 'z'")
#         sys.exit(1)
    
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     x_shift = float(sys.argv[3])
#     y_shift = float(sys.argv[4])
#     z_shift = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    
#     # Optional rotation parameters
#     rotation_angle = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
#     rotation_axis = sys.argv[7] if len(sys.argv) > 7 else 'z'
    
#     shift_pointcloud(input_file, output_file, x_shift, y_shift, z_shift, 
#                     rotation_angle, rotation_axis)



