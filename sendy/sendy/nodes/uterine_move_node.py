#!/usr/bin/env python3
"""
UterineMoveNode (left arm)
 
Executes the left-arm physical control sequence, driven by the manual state machine on
/current_task (task_publisher_node.py):
 
  - /current_task == "screw"            -> screw / insert 2 mm / screw at the current pose
  - /retract_start_action (PointCloud2) -> servo the tip along the trajectory
  - /retract_action        (PointCloud2) -> same
 
Trajectories run through the smoother_uterus visual servo (mover_uterus SmootherMover), closed on
the smoother's tracked tip. Nothing here uses motor_node's open-loop /fwkin_l or the IK node (/xyz_l);
don't run the IK node alongside the servo, since both would publish /robot/state/current_state.
 
Each trajectory is in hy/left/fwkin, planned by the diffusion node from the smoother's tracked tip. A
trajectory that starts far from where the tracked tip is now (the tip moved since planning) is refused.
 
The screw step uses /keyboard_control offsets (motor_node adds them to every command): two 3*pi rolls
of the inner tube, with a 2 mm push of the inner tube (along the tool) in between.
 
return_home_sequence() is kept but not wired to any task.
"""

import math
import time
 
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
 
from std_msgs.msg import Int32, Float32MultiArray, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener, TransformException
 
from mover_uterus.action_client import SmootherMover



# ============================================================================
# CONFIG
# ============================================================================

MIN_WAYPOINTS = 1
WAYPOINT_SETTLE_S = 0.5           # sleep after publishing each xyz target

SCREW_INSERT_DISTANCE_M = 0.002   # 2mm — see NOTE above (was 0.004 in move_node.py)
SCREW_ROLL_RAD = 3.0 * math.pi    # each screw roll of the inner tube
 
# Visual servo
TRAJ_FRAME = "hy/left/fwkin"               # diffusion model frame (fixed rotation of smoother_uterus/left/base)
TIP_FRAME = "smoother_uterus/left/tip"     # smoother's tracked tip (launch with left_tool_length:=0.0)
TIP_SPEED = 0.003                          # m/s along the trajectory
END_TOLERANCE = 0.002                      # m; farther than this from the last waypoint -> one settle goal
MAX_START_DISTANCE = 0.010                 # m; refuse a trajectory whose first point is farther from the tip
MAX_TIP_AGE_S = 1.0                        # s; refuse to run on an old tip (smoother not running)


CURRENT_TASK_TOPIC = "/current_task"
RETRACT_START_TOPIC = "/retract_start_action"
RETRACT_TOPIC = "/retract_action"
ROBOT_CMD_TOPIC = "/robot/state/current_state"

LEFT_INNER_ROT_CMD_IDX = 4        # command layout [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]
# /keyboard_control offsets use motor_node's motor layout [R_ot, R_or, R_it, R_ir, L_ot, L_or, L_it, L_ir]
KEYBOARD_LEFT_INNER_TRANS = 6
KEYBOARD_LEFT_INNER_ROT = 7


class UterineMoveNode(Node):
    def __init__(self):
        super().__init__('uterine_move_node')

        # Publishers
        # self.xyz_pub = self.create_publisher(Float32MultiArray, '/xyz_l', 10)
        self.keyboard_pub = self.create_publisher(Float32MultiArray, '/keyboard_control', 10)
        self.set_home_pub = self.create_publisher(Int32, '/set_home', 10)
        self.robot_pub = self.create_publisher(Float32MultiArray, ROBOT_CMD_TOPIC, 10)

        self.robot_pose = None

        # Subscribers
        self.task_sub = self.create_subscription(String, CURRENT_TASK_TOPIC, self.task_callback, 10)
        self.retract_start_sub = self.create_subscription(
            PointCloud2, RETRACT_START_TOPIC, self.trajectory_callback, 10)
        self.retract_sub = self.create_subscription(
            PointCloud2, RETRACT_TOPIC, self.trajectory_callback, 10)
        self.robot_sub = self.create_subscription(
            Float32MultiArray, ROBOT_CMD_TOPIC, self.robot_callback, 10)

        # Tracked tip from smoother_uterus, and the servo client (it waits on its own executor, so it
        # can block inside our callbacks)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.mover = SmootherMover('left_servo_client')


        self.get_logger().info(
            f'UterineMoveNode initialized. Listening on {CURRENT_TASK_TOPIC} for "screw", '
            f'and on {RETRACT_START_TOPIC} / {RETRACT_TOPIC} for trajectories.'
        )

    # ------------- Callbacks -------------

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
        Triggered on /retract_start_action or /retract_action. Servos the left tip through every point.
        Blocks until the servo finishes.
        """
        self.get_logger().info('Received left-arm trajectory.')
 
        frame = cloud_msg.header.frame_id or TRAJ_FRAME
        if frame != TRAJ_FRAME:
            self.get_logger().error(f'Trajectory is in "{frame}", expected "{TRAJ_FRAME}". Not running it.')
            return
 
        pts = [[float(x), float(y), float(z)] for x, y, z in
               point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)]
        if len(pts) < MIN_WAYPOINTS:
            self.get_logger().warn(f'Trajectory had {len(pts)} points, expected at least {MIN_WAYPOINTS}.')
            return
        waypoints = np.array(pts)
 
        roll = self.current_roll()
        if roll is None:
            return
 
        tip = self.tracked_tip()
        if tip is None:
            return
        start_distance = float(np.linalg.norm(waypoints[0] - tip))
        if start_distance > MAX_START_DISTANCE:
            self.get_logger().error(
                f'First waypoint is {start_distance * 1000:.1f} mm from the tracked tip (limit '
                f'{MAX_START_DISTANCE * 1000:.0f} mm); the tip moved since planning. Not running it.')
            return
 
        self.get_logger().info(
            f'Servoing {len(waypoints)}-point trajectory at {TIP_SPEED * 1000:.1f} mm/s, '
            f'end {np.round(waypoints[-1] * 1000, 1)} mm in {TRAJ_FRAME}')
        result = self.mover.servo_arm('left', waypoints.tolist(), TRAJ_FRAME,
                                      tool_rotation=roll, tip_speed=TIP_SPEED)
        error = self.end_error(result, waypoints[-1])
 
        # The servo stops when the trajectory's time runs out, not when the tip arrives
        if error is not None and error > END_TOLERANCE:
            self.get_logger().warn(
                f'Ended {error * 1000:.1f} mm from the last waypoint; sending one settle goal.')
            result = self.mover.servo_arm('left', [waypoints[-1].tolist()], TRAJ_FRAME,
                                          tool_rotation=roll, tip_speed=TIP_SPEED)
            error = self.end_error(result, waypoints[-1])
 
        if error is None:
            self.get_logger().error('Trajectory did not run to the end (goal rejected or aborted, see above).')
        else:
            self.get_logger().info(f'Trajectory done, {error * 1000:.1f} mm from the last waypoint.')


    # def trajectory_callback(self, cloud_msg: PointCloud2):
    #     """
    #     Triggered on /retract_start_action or /retract_action. Walks through
    #     every point in the message via /xyz_l, waiting to reach each one.
    #     Pure point-to-point motion — no screw sequence, no return-home.
    #     """
    #     self.get_logger().info('Received left-arm trajectory.')

    #     # pts = list(
    #     #     point_cloud2.read_points(
    #     #         cloud_msg,
    #     #         field_names=('x', 'y', 'z'),
    #     #         skip_nans=True
    #     #     )
    #     # )
    #     pts = [[float(x), float(y), float(z)] for x, y, z in point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)]


    #     if len(pts) < MIN_WAYPOINTS:
    #         self.get_logger().warn(
    #             f'Trajectory had {len(pts)} points, expected at least {MIN_WAYPOINTS}.'
    #         )
    #         return
    #     waypoints = np.array(pts)
    #     roll = self.current_roll()
    #     if roll is None:
    #         return

    #     self.get_logger().info(f'Executing {len(pts)}-point trajectory.')

    #     for idx, (x, y, z) in enumerate(pts):
    #         self.get_logger().info(f'Moving to point {idx + 1}/{len(pts)}: ({x:.4f}, {y:.4f}, {z:.4f})')
    #         self.publish_xyz(x, y, z)
    #         time.sleep(WAYPOINT_SETTLE_S)
    #         self.wait_until_reached(x, y, z)

    #     self.get_logger().info('Trajectory execution done.')

    # ------------- Helper methods -------------

    def current_roll(self):
        """Left inner-tube rotation of the last command, so the servo holds it (it sets that joint directly)."""
        if self.robot_pose is None or len(self.robot_pose.data) < 8:
            self.get_logger().error(
                f'No command seen on {ROBOT_CMD_TOPIC} yet, so the tool roll to hold is unknown. '
                f'Send one first (e.g. go home).')
            return None
        return float(self.robot_pose.data[LEFT_INNER_ROT_CMD_IDX])
 
    def tracked_tip(self):
        """The smoother's tracked left tip in TRAJ_FRAME, or None if it isn't available or is old."""
        try:
            t = self.tf_buffer.lookup_transform(TRAJ_FRAME, TIP_FRAME, Time())
        except TransformException as e:
            self.get_logger().error(f'No tracked tip ({TIP_FRAME} in {TRAJ_FRAME}): {e}. Is smoother_uterus running?')
            return None
 
        age = (self.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds / 1e9
        if age > MAX_TIP_AGE_S:
            self.get_logger().error(f'Tracked tip is {age:.1f} s old; is smoother_uterus running?')
            return None
 
        p = t.transform.translation
        return np.array([p.x, p.y, p.z])
 
    def end_error(self, result, last_waypoint):
        """Distance between the servo's last measured tip and the last waypoint, or None if it didn't run."""
        if result is None or not result.success or len(result.left_measured_traj) == 0:
            return None
        m = result.left_measured_traj[-1]
        return float(np.linalg.norm(np.array([m.x, m.y, m.z]) - last_waypoint))


    def publish_keyboard_command(self, data_list):
        msg = Float32MultiArray()
        msg.data = list(map(float, data_list))
        self.keyboard_pub.publish(msg)

    def apply_offset(self, index, value):
        """Add a /keyboard_control offset, then republish the last command so motor_node applies it."""
        cmd = [0.0] * 8
        cmd[index] = value
        self.publish_keyboard_command(cmd)
        self.get_logger().info(f'Published keyboard command: {cmd}')
        self.robot_pub.publish(self.robot_pose)


    def screw_sequence(self):
        """
        Screw / insert 2 mm / screw at the current pose, all as keyboard offsets on the left inner tube.
        The insert pushes the inner tube, i.e. along the tool. (The old IK version moved 2 mm along the
        arm's base axis instead, which isn't the tool direction once the outer tube bends.)
        """
        if self.robot_pose is None:
            self.get_logger().error(
                f'No command seen on {ROBOT_CMD_TOPIC} yet; offsets would not be applied. Send one first.')
            return
 
        self.get_logger().info('Executing screw sequence at current position.')
 
        self.apply_offset(KEYBOARD_LEFT_INNER_ROT, SCREW_ROLL_RAD)
        time.sleep(WAYPOINT_SETTLE_S)
 
        self.apply_offset(KEYBOARD_LEFT_INNER_TRANS, SCREW_INSERT_DISTANCE_M)
        time.sleep(WAYPOINT_SETTLE_S)
 
        self.apply_offset(KEYBOARD_LEFT_INNER_ROT, SCREW_ROLL_RAD)
 
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
        node.mover.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
