#!/usr/bin/env python3
"""
UterineMoveNode (left arm)

Executes the left-arm physical control sequence, now driven by the manual
state machine on /current_task (task_publisher_node.py) instead of running
its own keyboard loop:

  - /current_task == "screw"          -> run the screw / insert / screw
                                          sequence directly, using the
                                          CURRENT tool pose (/fwkin_l) as the
                                          reference point (no trajectory
                                          needed).
  - /retract_start_action (PointCloud2) -> walk every waypoint via /xyz_l,
                                          waiting to reach each one.
  - /retract_action        (PointCloud2) -> same waypoint walk.

This is a behavior change from the original move_node.py: the screw sequence
used to be embedded inside the trajectory walk (triggered automatically at
waypoint 0), and the node blocked on input() waiting for ENTER before running
a return-home sequence. Both of those are gone:
  - screw is now its own explicit step in the sequence (operator types
    "screw" via task_publisher_node when they want it to run), so retract /
    retract_start trajectories are now PURE point-to-point moves.
  - there's no automatic return-home anymore — return_home_sequence() is
    kept below as a ready-to-wire method, but nothing currently calls it.
    Blocking on input() inside a subscription callback would freeze this
    node's ability to react to /current_task, which no longer fits the
    external-state-machine model, so it's been removed rather than kept
    as a hidden hazard. Wire it to a future "home_left" task when you want it.

NOTE: the screw sequence's insertion distance was changed from the original
move_node.py's 0.004 m (4mm) to 0.002 m (2mm) to match "screw then 2mm then
screw" — double check SCREW_INSERT_DISTANCE_M below if that's not what you
meant.
"""

import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32, Float32MultiArray, String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np


# ============================================================================
# CONFIG
# ============================================================================

MIN_WAYPOINTS = 1
POSITION_TOLERANCE = 1000e-2      # meters, same as move_node.py
WAYPOINT_SETTLE_S = 0.5           # sleep after publishing each xyz target

SCREW_INSERT_DISTANCE_M = 0.002   # 2mm — see NOTE above (was 0.004 in move_node.py)

CURRENT_TASK_TOPIC = "/current_task"
RETRACT_START_TOPIC = "/retract_start_action"
RETRACT_TOPIC = "/retract_action"


class UterineMoveNode(Node):
    def __init__(self):
        super().__init__('uterine_move_node')

        # Publishers
        self.xyz_pub = self.create_publisher(Float32MultiArray, '/xyz_l', 10)
        self.keyboard_pub = self.create_publisher(Float32MultiArray, '/keyboard_control', 10)
        self.set_home_pub = self.create_publisher(Int32, '/set_home', 10)
        self.robot_pub = self.create_publisher(Float32MultiArray, '/robot/state/current_state', 10)

        self.robot_pose = None

        # Subscribers
        self.task_sub = self.create_subscription(String, CURRENT_TASK_TOPIC, self.task_callback, 10)
        self.retract_start_sub = self.create_subscription(
            PointCloud2, RETRACT_START_TOPIC, self.trajectory_callback, 10)
        self.retract_sub = self.create_subscription(
            PointCloud2, RETRACT_TOPIC, self.trajectory_callback, 10)
        self.fwkin_sub = self.create_subscription(Pose, '/fwkin_l', self.fwkin_callback, 10)
        self.robot_sub = self.create_subscription(
            Float32MultiArray, '/robot/state/current_state', self.robot_callback, 10)

        self.current_pose = None

        self.get_logger().info(
            f'UterineMoveNode initialized. Listening on {CURRENT_TASK_TOPIC} for "screw", '
            f'and on {RETRACT_START_TOPIC} / {RETRACT_TOPIC} for trajectories.'
        )

    # ------------- Callbacks -------------

    def fwkin_callback(self, msg: Pose):
        """Store latest FK pose from /fwkin_l."""
        self.current_pose = msg

    def robot_callback(self, msg: Float32MultiArray):
        """Store latest robot pose from /robot/state/current_state."""
        self.robot_pose = msg

    def task_callback(self, msg: String):
        task = msg.data.strip()
        if task == "screw":
            self.screw_sequence()
        # retract_start / retract / resect* are handled via their own topics,
        # not this callback — nothing else to do here.

    def trajectory_callback(self, cloud_msg: PointCloud2):
        """
        Triggered on /retract_start_action or /retract_action. Walks through
        every point in the message via /xyz_l, waiting to reach each one.
        Pure point-to-point motion — no screw sequence, no return-home.
        """
        self.get_logger().info('Received left-arm trajectory.')

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
        self.get_logger().info(f'Published target to /xyz_l: ({x:.4f}, {y:.4f}, {z:.4f})')

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

    def publish_keyboard_command(self, data_list):
        msg = Float32MultiArray()
        msg.data = list(map(float, data_list))
        self.keyboard_pub.publish(msg)

    def screw_sequence(self):
        """
        Screw / insert 2mm / screw, run at the CURRENT tool position
        (/fwkin_l) rather than a trajectory waypoint, since "screw" is now
        its own independently-triggered task instead of being embedded in
        the retract trajectory walk.
        """
        if self.current_pose is None:
            self.get_logger().error('No /fwkin_l pose yet; cannot run screw sequence.')
            return

        ref = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z,
        ])

        self.get_logger().info('Executing screw sequence at current position.')

        cmd1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0 * math.pi]
        self.publish_keyboard_command(cmd1)
        self.get_logger().info(f'Published keyboard command: {cmd1}')
        if self.robot_pose is not None:
            self.robot_pub.publish(self.robot_pose)
        time.sleep(WAYPOINT_SETTLE_S)

        self.publish_xyz(ref[0], ref[1], ref[2] + SCREW_INSERT_DISTANCE_M)
        time.sleep(WAYPOINT_SETTLE_S)

        cmd3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0 * math.pi]
        self.publish_keyboard_command(cmd3)
        self.get_logger().info(f'Published keyboard command: {cmd3}')
        if self.robot_pose is not None:
            self.robot_pub.publish(self.robot_pose)

        self.get_logger().info('Screw sequence done.')

    def return_home_sequence(self):
        """Not currently wired to any task — see module docstring. Kept for
        when a "home_left" task is added to task_publisher_node.py."""
        self.get_logger().info('Executing return-home sequence...')

        cmd_back = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0 * math.pi]
        self.publish_keyboard_command(cmd_back)
        self.get_logger().info(f'Published keyboard command: {cmd_back}')
        if self.robot_pose is not None:
            self.robot_pub.publish(self.robot_pose)
        time.sleep(WAYPOINT_SETTLE_S)

        home_msg = Int32()
        home_msg.data = 1
        self.set_home_pub.publish(home_msg)
        self.get_logger().info('Published 1 to /set_home.')


def main(args=None):
    rclpy.init(args=args)
    node = UterineMoveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
