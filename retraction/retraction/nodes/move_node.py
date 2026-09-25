#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32, Float32MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np


class MoveNode(Node):
    def __init__(self):
        super().__init__('move_node')

        # Publishers
        self.xyz_pub = self.create_publisher(Float32MultiArray, '/xyz_l', 10)
        self.keyboard_pub = self.create_publisher(Float32MultiArray, '/keyboard_control', 10)
        self.set_home_pub = self.create_publisher(Int32, '/set_home', 10)
        self.robot_pub = self.create_publisher(Float32MultiArray, '/robot/state/current_state', 10)

        self.start_pt = None

        # Subscribers
        self.retract_sub = self.create_subscription(
            PointCloud2,
            '/retract_action',
            self.retract_callback,
            10
        )
        self.fwkin_sub = self.create_subscription(
            Pose,
            '/fwkin_l',
            self.fwkin_callback,
            10
        )
        self.robot_sub = self.create_subscription(
            Float32MultiArray,
            '/robot/state/current_state',
            self.robot_callback,
            10
        )

        # Store latest forward kinematics pose
        self.current_pose = None

        # Distance tolerance for "reached point"
        self.position_tolerance = 1000e-3  # meters

        self.get_logger().info('RetractActionNode initialized.')

    # ------------- Callbacks -------------

    def fwkin_callback(self, msg: Pose):
        """Store latest FK pose from /fwkin_l."""
        print("current pose", self.current_pose)
        self.current_pose = msg

    def robot_callback(self, msg: Float32MultiArray):
        """Store latest robot pose from /robot/state/current_state"""
        self.robot_pose = msg
        print("robot pose", self.robot_pose)


    def retract_callback(self, cloud_msg: PointCloud2):
        """
        Triggered when a PointCloud2 with 2 points is received on /retract_action.
        Follows the sequence for each point, then waits for Enter, then sends home.
        """
        self.get_logger().info('Received /retract_action message.')

        # Extract points from PointCloud2
        pts = list(
            point_cloud2.read_points(
                cloud_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )
        )

        if len(pts) < 2:
            self.get_logger().warn(
                f'/retract_action had {len(pts)} points, expected at least 2.'
            )
            return

        # Only use the first 2 points as requested
        target_points = pts[:2]
        print(target_points, "TARGET PTS")
        for idx, (x, y, z) in enumerate(target_points):
            self.get_logger().info(f'Processing point {idx + 1}: ({x:.4f}, {y:.4f}, {z:.4f})')

            # 1) Publish target point to /xyz_l
            if idx == 0: 
                self.start_pt = np.array([x,y,z])
            self.publish_xyz(x, y, z)

            time.sleep(5.0)

            # 2) Wait until /fwkin_l matches this point (within tolerance)
            self.wait_until_reached(x, y, z)
            if idx == 0:
            # 3) Execute the keyboard control sequence
                self.execute_keyboard_sequence()

        # After both points:
        # 4) Wait until Enter key is pressed
        self.get_logger().info('Retraction steps done. Press ENTER to proceed with return-home sequence...')
        input()  # Blocks until user presses Enter in the terminal

        # 5) Return-home sequence: send [-4*pi] on 7th element, wait 3s, then /set_home = 1
        self.return_home_sequence()

        self.get_logger().info('Sequence complete. Waiting for next /retract_action message.')

    # ------------- Helper methods -------------

    def publish_xyz(self, x: float, y: float, z: float):
        """Publish target xyz point to /xyz_l as Float32MultiArray [x, y, z]."""
        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(z)]
        self.xyz_pub.publish(msg)
        self.get_logger().info(
            f'Published target to /xyz_l: ({x:.4f}, {y:.4f}, {z:.4f})'
        )

    def wait_until_reached(self, x: float, y: float, z: float, timeout: float = None):
        """
        Busy-wait (with sleep) until current_pose.position matches (x, y, z)
        within self.position_tolerance. Optional timeout (seconds).
        """
        self.get_logger().info('Waiting for FK pose to reach target point...')
        start_time = time.time()
        print("current pose: ",self.current_pose)
        while rclpy.ok():
            if self.current_pose is not None:
                dx = self.current_pose.position.x - x
                dy = self.current_pose.position.y - y
                dz = self.current_pose.position.z - z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                print(dist)


                if dist <= self.position_tolerance:
                    self.get_logger().info(
                        f'Reached target point. Distance: {dist:.6f} m'
                    )
                    return

            if timeout is not None and (time.time() - start_time) > timeout:
                self.get_logger().warn(
                    f'Timeout while waiting to reach target point ({x}, {y}, {z}).'
                )
                return

            time.sleep(0.01)  # small sleep to avoid busy spin

    def publish_keyboard_command(self, data_list):
        msg = Float32MultiArray()
        msg.data = list(map(float, data_list))
        self.keyboard_pub.publish(msg)

    def execute_keyboard_sequence(self):
        """
        For each point in /retract_action:
          - publish [0 0 0 0 0 0 4*pi 0] to /keyboard_control
          - wait 2 sec
          - publish [0 0 0 0 0 0.005 0 0]
          - wait 2 sec
          - publish [0 0 0 0 0 0 4*pi 0]
        """
        self.get_logger().info('Executing keyboard control sequence for this point.')

        # 1) [0, 0, 0, 0, 0, 0, 4*M_PI, 0]
        cmd1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0 * math.pi]
        self.publish_keyboard_command(cmd1)
        self.get_logger().info(f'Published keyboard command: {cmd1}')
        self.robot_pub.publish(self.robot_pose)
        print("SCREW1",self.robot_pose)
        time.sleep(5.0)


        # move a bit in x
        self.publish_xyz(self.start_pt[0],self.start_pt[1], self.start_pt[2] + 0.004)

        time.sleep(5.0)

        # # 2) [0, 0, 0, 0, 0, 0.005, 0, 0]
        # cmd2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.0]
        # self.publish_keyboard_command(cmd2)
        # self.get_logger().info(f'Published keyboard command: {cmd2}')
        # self.robot_pub.publish(self.robot_pose)
        # print("INSERT",self.robot_pose)
        # time.sleep(5.0)

        # 3) [0, 0, 0, 0, 0, 0, 4*M_PI, 0]
        cmd3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 * math.pi]
        self.publish_keyboard_command(cmd3)
        self.get_logger().info(f'Published keyboard command: {cmd3}')
        self.robot_pub.publish(self.robot_pose)
        print("SCREW2",self.robot_pose)

    def return_home_sequence(self):
        """
        After Enter is pressed:
          - publish [0 0 0 0 0 0 -4*M_PI 0] to /keyboard_control
          - wait 3 seconds
          - publish 1 to /set_home
        """
        self.get_logger().info('Executing return-home sequence...')

        cmd_back = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0 * math.pi]
        self.publish_keyboard_command(cmd_back)
        self.get_logger().info(f'Published keyboard command: {cmd_back}')
        self.robot_pub.publish(self.robot_pose)
        time.sleep(5.0)

        home_msg = Int32()
        home_msg.data = 1
        self.set_home_pub.publish(home_msg)
        self.get_logger().info('Published 1 to /set_home.')

    # --------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = MoveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
