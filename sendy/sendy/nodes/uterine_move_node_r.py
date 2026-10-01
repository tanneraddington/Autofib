#!/usr/bin/env python3
"""
UterineMoveNodeR (right arm — cautery/resection)

Servos the right-arm tip along the resection trajectories published by uterine_diffusion_node:

  - /resect_start_action (PointCloud2) -> servo the tip along the trajectory
  - /resect_action        (PointCloud2) -> same

Trajectories run through the smoother_uterus visual servo (mover_uterus SmootherMover), closed on
the smoother's tracked tip. Nothing here uses motor_node's open-loop /fwkin_r or the IK node (/xyz_r);
don't run the IK node alongside the servo, since both would publish /robot/state/current_state.

Each trajectory is in hy/right/fwkin, planned by the diffusion node from the smoother's tracked tip. A
trajectory that starts far from where the tracked tip is now (the tip moved since planning) is refused.

Cautery execution is plain point-to-point motion: no screw/insert sequence and no energy-activation
trigger (cautery isn't wired to the servo on this robot), so unlike the left arm there's no
/current_task step here, just the two trajectory topics.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener, TransformException

from mover_uterus.action_client import SmootherMover


# ============================================================================
# CONFIG
# ============================================================================

MIN_WAYPOINTS = 1

# Visual servo
TRAJ_FRAME = "hy/right/fwkin"              # diffusion model frame (fixed rotation of smoother_uterus/right/base)
TIP_FRAME = "smoother_uterus/right/tip"    # smoother's tracked tip (launch with right_tool_length:=0.0)
TIP_SPEED = 0.003                          # m/s along the trajectory
END_TOLERANCE = 0.002                      # m; farther than this from the last waypoint -> one settle goal
MAX_START_DISTANCE = 0.010                 # m; refuse a trajectory whose first point is farther from the tip
MAX_TIP_AGE_S = 1.0                        # s; refuse to run on an old tip (smoother not running)

RESECT_START_TOPIC = "/resect_start_action"
RESECT_TOPIC = "/resect_action"
ROBOT_CMD_TOPIC = "/robot/state/current_state"

RIGHT_INNER_ROT_CMD_IDX = 0       # command layout [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]


class UterineMoveNodeR(Node):
    def __init__(self):
        super().__init__('uterine_move_node_r')

        self.robot_pose = None      # last command on /robot/state/current_state (for the tool roll)

        # Subscribers
        self.resect_start_sub = self.create_subscription(
            PointCloud2, RESECT_START_TOPIC, self.trajectory_callback, 10)
        self.resect_sub = self.create_subscription(
            PointCloud2, RESECT_TOPIC, self.trajectory_callback, 10)
        self.robot_sub = self.create_subscription(
            Float32MultiArray, ROBOT_CMD_TOPIC, self.robot_callback, 10)

        # Tracked tip from smoother_uterus, and the servo client (it waits on its own executor, so it
        # can block inside our callbacks)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.mover = SmootherMover('right_servo_client')

        self.get_logger().info(
            f'UterineMoveNodeR initialized. Listening on {RESECT_START_TOPIC} / {RESECT_TOPIC} '
            f'for trajectories (visual servo).'
        )

    # ------------- Callbacks -------------

    def robot_callback(self, msg: Float32MultiArray):
        """Store latest command from /robot/state/current_state."""
        self.robot_pose = msg

    def trajectory_callback(self, cloud_msg: PointCloud2):
        """
        Triggered on /resect_start_action or /resect_action. Servos the right tip through every point.
        Blocks until the servo finishes. Plain point-to-point motion: no screw, no energy trigger.
        """
        self.get_logger().info('Received right-arm (cautery) trajectory.')

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
        result = self.mover.servo_arm('right', waypoints.tolist(), TRAJ_FRAME,
                                      tool_rotation=roll, tip_speed=TIP_SPEED)
        error = self.end_error(result, waypoints[-1])

        # The servo stops when the trajectory's time runs out, not when the tip arrives
        if error is not None and error > END_TOLERANCE:
            self.get_logger().warn(
                f'Ended {error * 1000:.1f} mm from the last waypoint; sending one settle goal.')
            result = self.mover.servo_arm('right', [waypoints[-1].tolist()], TRAJ_FRAME,
                                          tool_rotation=roll, tip_speed=TIP_SPEED)
            error = self.end_error(result, waypoints[-1])

        if error is None:
            self.get_logger().error('Trajectory did not run to the end (goal rejected or aborted, see above).')
        else:
            self.get_logger().info(f'Trajectory done, {error * 1000:.1f} mm from the last waypoint.')

    # ------------- Helper methods -------------

    def current_roll(self):
        """Right inner-tube rotation of the last command, so the servo holds it (it sets that joint directly)."""
        if self.robot_pose is None or len(self.robot_pose.data) < 8:
            self.get_logger().error(
                f'No command seen on {ROBOT_CMD_TOPIC} yet, so the tool roll to hold is unknown. '
                f'Send one first (e.g. go home).')
            return None
        return float(self.robot_pose.data[RIGHT_INNER_ROT_CMD_IDX])

    def tracked_tip(self):
        """The smoother's tracked right tip in TRAJ_FRAME, or None if it isn't available or is old."""
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
        if result is None or not result.success or len(result.right_measured_traj) == 0:
            return None
        m = result.right_measured_traj[-1]
        return float(np.linalg.norm(np.array([m.x, m.y, m.z]) - last_waypoint))


def main(args=None):
    rclpy.init(args=args)
    node = UterineMoveNodeR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.mover.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()