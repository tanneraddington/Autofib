"""
Minimal PointCloud2 <-> numpy helpers + image preprocessing utility, trimmed
from diffusion_displacement/point_cloud_utils for the uterine system nodes.

Kept intentionally small (no open3d / sklearn dependency) since the uterine
nodes only ever move flat [N, 3] xyz arrays around.
"""

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def pointcloud2_to_xyz_array(cloud_msg: PointCloud2) -> np.ndarray:
    """Convert a PointCloud2 message to an [N, 3] numpy array of xyz points."""
    return point_cloud2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)


def xyz_array_to_pointcloud2(points: np.ndarray, frame_id: str = "hy/base") -> PointCloud2:
    """Convert an [N, 3] numpy array of xyz points to a PointCloud2 message."""
    header = Header()
    header.frame_id = frame_id

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    return point_cloud2.create_cloud(header, fields, points)


GRAY_VALUE = 128


def fill_endoscope_border(image: np.ndarray, gray_value: int = GRAY_VALUE,
                           protect_radius_frac: float = 0.35, black_thresh: int = 20) -> np.ndarray:
    """
    Replace near-black endoscope border pixels with gray, except inside a
    protected center region where true dark content (tissue/instruments) may
    exist. Matches the preprocessing used to train the displacement_diffusion
    models, so inference-time images stay in-distribution.
    """
    if image is None:
        return None

    img = image.copy()
    h, w = img.shape[:2]

    is_black = np.all(img <= black_thresh, axis=2)

    cy, cx = h // 2, w // 2
    protect_radius = int(min(h, w) * protect_radius_frac)
    Y, X = np.ogrid[:h, :w]
    in_center = ((X - cx) ** 2 + (Y - cy) ** 2) <= protect_radius ** 2

    fill_mask = is_black & ~in_center
    img[fill_mask] = gray_value

    return img
