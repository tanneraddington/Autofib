#!/usr/bin/env python3
"""
UterineMoveNodeR (right arm — cautery/resection)

Mirrors uterine_move_node.py's waypoint-walking logic for the right arm:
subscribes to the resection trajectories published by uterine_diffusion_node
and walks each waypoint via /xyz_r, using /fwkin_r for reach-confirmation
feedback.

Per confirmation: cautery execution is plain point-to-point motion, no
screw/insert sequence and no separate energy-activation trigger — so unlike
the left arm, there's no /current_task-triggered step here, just the two
trajectory topics.

  - /resect_start_action (PointCloud2) -> walk waypoints via /xyz_r
  - /resect_action        (PointCloud2) -> walk waypoints via /xyz_r
"""

import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


# ============================================================================
# CONFIG
# ============================================================================

MIN_WAYPOINTS = 1
POSITION_TOLERANCE = 1000e-3      # meters, matches uterine_move_node.py
WAYPOINT_SETTLE_S = 1.0

RESECT_START_TOPIC = "/resect_start_action"
RESECT_TOPIC = "/resect_action"


class UterineMoveNodeR(Node):
    def __init__(self):
        super().__init__('uterine_move_node_r')

        self.xyz_pub = self.create_publisher(Float32MultiArray, '/xyz_r', 10)

        self.resect_start_sub = self.create_subscription(
            PointCloud2, RESECT_START_TOPIC, self.trajectory_callback, 10)
        self.resect_sub = self.create_subscription(
            PointCloud2, RESECT_TOPIC, self.trajectory_callback, 10)
        self.fwkin_sub = self.create_subscription(Pose, '/fwkin_r', self.fwkin_callback, 10)

        self.current_pose = None

        self.get_logger().info(
            f'UterineMoveNodeR initialized. Listening on {RESECT_START_TOPIC} / {RESECT_TOPIC}.'
        )

    # ------------- Callbacks -------------

    def fwkin_callback(self, msg: Pose):
        """Store latest FK pose from /fwkin_r."""
        self.current_pose = msg

    def trajectory_callback(self, cloud_msg: PointCloud2):
        """
        Triggered on /resect_start_action or /resect_action. Walks through
        every point in the message via /xyz_r, waiting to reach each one.
        Plain point-to-point motion — no screw, no energy trigger.
        """
        self.get_logger().info('Received right-arm (cautery) trajectory.')

        pts = list(
            point_cloud2.read_points(
                cloud_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )
        )

        if len(pts) < MIN_WAYPOINTS:
            self.get_logger().warn(
                f'Trajectory had {len(pts)} points, expected at least {MIN_WAYPOINTS}.'
            )
            return

        self.get_logger().info(f'Executing {len(pts)}-point trajectory.')

        for idx, (x, y, z) in enumerate(pts):
            self.get_logger().info(f'Moving to point {idx + 1}/{len(pts)}: ({x:.4f}, {y:.4f}, {z:.4f})')
            self.publish_xyz(x, y, z)
            time.sleep(WAYPOINT_SETTLE_S)
            self.wait_until_reached(x, y, z)

        self.get_logger().info('Trajectory execution done.')

    # ------------- Helper methods -------------

    def publish_xyz(self, x: float, y: float, z: float):
        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(z)]
        self.xyz_pub.publish(msg)
        self.get_logger().info(f'Published target to /xyz_r: ({x:.4f}, {y:.4f}, {z:.4f})')

    def wait_until_reached(self, x: float, y: float, z: float, timeout: float = None):
        self.get_logger().info('Waiting for FK pose to reach target point...')
        start_time = time.time()
        while rclpy.ok():
            if self.current_pose is not None:
                dx = self.current_pose.position.x - x
                dy = self.current_pose.position.y - y
                dz = self.current_pose.position.z - z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist <= POSITION_TOLERANCE:
                    self.get_logger().info(f'Reached target point. Distance: {dist:.6f} m')
                    return

            if timeout is not None and (time.time() - start_time) > timeout:
                self.get_logger().warn(f'Timeout while waiting to reach target point ({x}, {y}, {z}).')
                return

            time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args)
    node = UterineMoveNodeR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
