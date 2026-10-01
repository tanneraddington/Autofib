#!/usr/bin/env python3
"""
Rectifies /image with a camera_calibration YAML (ost.yaml) and publishes what smoother_uterus expects:
  /camera/image_rect   sensor_msgs/Image       undistorted, same encoding and header as /image
  /camera/camera_info  sensor_msgs/CameraInfo  from the YAML, same header as each frame
Replaces the CameraInfo + image_proc steps for a camera node that can't load a calibration file.

Run:  python3 camera_rectify_node.py
With these topics the smoother launch needs no image_topic / camera_info_topic arguments.
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

CALIBRATION_YAML = os.path.expanduser('~/calibration/ost.yaml')  # from /tmp/calibrationdata.tar.gz

# How much of the original view to keep after undistortion (cv2.getOptimalNewCameraMatrix alpha):
#   0.0 = crop to only valid pixels (what camera_calibration's P does: zoomed in, no black borders)
#   1.0 = keep every source pixel (full view, curved black borders at the edges)
RECTIFY_ALPHA = 1.0

IMAGE_TOPIC = '/image'
IMAGE_RECT_TOPIC = '/camera/image_rect'
CAMERA_INFO_TOPIC = '/camera/camera_info'


def load_camera_info(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    info = CameraInfo()
    info.width = int(c['image_width'])
    info.height = int(c['image_height'])
    info.distortion_model = c['distortion_model']
    info.d = [float(v) for v in c['distortion_coefficients']['data']]
    info.k = [float(v) for v in c['camera_matrix']['data']]
    info.r = [float(v) for v in c['rectification_matrix']['data']]
    info.p = [float(v) for v in c['projection_matrix']['data']]
    return info


class CameraRectifyNode(Node):
    def __init__(self):
        super().__init__('camera_rectify_node')
        self.bridge = CvBridge()
        self.info = load_camera_info(CALIBRATION_YAML)

        K = np.array(self.info.k).reshape(3, 3)
        D = np.array(self.info.d)
        R = np.array(self.info.r).reshape(3, 3)
        size = (self.info.width, self.info.height)

        # Pick the output camera matrix ourselves instead of using the YAML's P (which is cropped)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, RECTIFY_ALPHA, size)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(K, D, R, new_K, size, cv2.CV_32FC1)

        # Publish the P that matches the image we output: the smoother projects with P
        P = np.zeros((3, 4))
        P[:, :3] = new_K
        self.info.p = P.flatten().tolist()

        self.image_pub = self.create_publisher(Image, IMAGE_RECT_TOPIC, 1)
        self.info_pub = self.create_publisher(CameraInfo, CAMERA_INFO_TOPIC, 1)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 1)

        self.get_logger().info(
            f'Rectifying {IMAGE_TOPIC} ({size[0]}x{size[1]}) with {CALIBRATION_YAML} -> '
            f'{IMAGE_RECT_TOPIC}, {CAMERA_INFO_TOPIC} | alpha {RECTIFY_ALPHA}: '
            f'fx={new_K[0, 0]:.1f} fy={new_K[1, 1]:.1f} cx={new_K[0, 2]:.1f} cy={new_K[1, 2]:.1f}')

    def image_callback(self, msg):
        if (msg.width, msg.height) != (self.info.width, self.info.height):
            self.get_logger().error(
                f'/image is {msg.width}x{msg.height} but the calibration is '
                f'{self.info.width}x{self.info.height}; recalibrate at this resolution',
                throttle_duration_sec=5.0)
            return

        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rect = cv2.remap(raw, self.map_x, self.map_y, cv2.INTER_LINEAR)

        out = self.bridge.cv2_to_imgmsg(rect, encoding=msg.encoding)
        out.header = msg.header
        self.info.header = msg.header

        self.image_pub.publish(out)
        self.info_pub.publish(self.info)


def main():
    rclpy.init()
    node = CameraRectifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()