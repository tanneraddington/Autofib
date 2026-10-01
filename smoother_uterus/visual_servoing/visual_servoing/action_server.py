import time
from enum import Enum
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.node import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Float64MultiArray, Float64, Float32MultiArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # <-- registers PointStamped/PoseStamped with TF2

from aliss_ros_msg.action import VisualServo
from aliss_ros_msg.msg import TouchSensor, VisualServoWaypoint


class ArmSide(Enum):
    LEFT = 0
    RIGHT = 1


arm_labels = {
    ArmSide.LEFT: "left",
    ArmSide.RIGHT: "right"
}

# ---------------------------------------------------------------------------
# Uterine robot interface (motor_node.cpp)
# ---------------------------------------------------------------------------
# Joint commands for BOTH arms go out as one Float32MultiArray on ROBOT_CMD_TOPIC:
#   [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]   (rad / m, relative to the motor_node home)
# Joint feedback per arm comes from motor_node on /robot/<side>/joint/measured_jp as
#   [inner_rot, outer_rot, inner_trans, outer_trans]; these include keyboard offsets.
ROBOT_CMD_TOPIC = '/robot/state/current_state'
ROBOT_CMD_SLICE = {
    ArmSide.RIGHT: slice(0, 4),
    ArmSide.LEFT: slice(4, 8),
}
MEASURED_JP_TOPIC = '/robot/{side}/joint/measured_jp'
SERVO_TOOL_TOPIC = '/robot/{side}/servo_tool'  # no subscriber on the uterine robot yet (no cautery interface)

# Tube extension limits. motor_node hard limits are outer <= 0.04 and inner <= 0.09 (plus keyboard offset).
INNER_EXTENSION_MAX = 0.045   # 60 mm tools
OUTER_EXTENSION_MAX = 0.04


def clip_joint_values(q):
    extension_min = 0.0
    inner_extension_max = INNER_EXTENSION_MAX
    outer_extension_max = OUTER_EXTENSION_MAX

    # Clip extensions
    q[2] = np.clip(q[2], extension_min, inner_extension_max)
    q[3] = np.clip(q[3], extension_min, outer_extension_max)

    # Clip outer extension at inner extension length
    q[3] = np.clip(q[3], extension_min, q[2])
    
    return q


def clip_joint_speed(q_dot):
    rotation_max = np.deg2rad(30)
    translation_max = 0.01

    q_dot[0] = np.clip(q_dot[0], -rotation_max, rotation_max)
    q_dot[1] = np.clip(q_dot[1], -rotation_max, rotation_max)
    q_dot[2] = np.clip(q_dot[2], -translation_max, translation_max)
    q_dot[3] = np.clip(q_dot[3], -translation_max, translation_max)

    return q_dot


class VisualServoServer(Node):
    def __init__(self):
        super().__init__('visual_servo_server')

        # Controller gains (camera-frame tip velocity)
        self.declare_parameter('vs_kp', 1.0)
        self.declare_parameter('vs_kff', 0.8)
        self.vs_kp = float(self.get_parameter('vs_kp').value)
        self.vs_kff = float(self.get_parameter('vs_kff').value)

        # Cautery flicker control (waypoint-based when interval > 0)
        self.declare_parameter('cautery_flicker_interval_waypoints', 0)
        self.declare_parameter('cautery_flicker_off_duration_sec', 0.05)
        self.cautery_flicker_interval_waypoints = int(
            self.get_parameter('cautery_flicker_interval_waypoints').value
        )
        self.cautery_flicker_off_duration_sec = float(
            self.get_parameter('cautery_flicker_off_duration_sec').value
        )
        self.add_on_set_parameters_callback(self.on_parameter_update)

        # TF stuff for getting TFs and transforming points
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Get camera info for projecting points to pixels and vise versa
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback, 
            10
        )

        # Set when camera info is received
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        # Need touch sensor input for stopping on touch, if requested
        self.create_subscription(
            TouchSensor, 
            "/touch_sensor/sensor_output",
            self.touch_sensing_callback, 
            10
        )

        # Default to false, so we can still run without touch sensor being published
        self.is_touching = False

        # Commanded joint state publisher (both arms in one message, see ROBOT_CMD_TOPIC)
        self.robot_cmd_pub = self.create_publisher(Float32MultiArray, ROBOT_CMD_TOPIC, 1)

        # Track the last command anyone sent, so we start from (and hold the idle arm at) the robot's
        # current COMMAND values. Measured joints include keyboard offsets, which motor_node adds on
        # top of the command, so seeding from them would jump the arm by that offset.
        self.last_robot_cmd = None
        self.create_subscription(Float32MultiArray, ROBOT_CMD_TOPIC, self.robot_cmd_callback, 10)

        # We need to also subscribe to the current joint state to init joint commands
        joint_state_callbacks = {
            ArmSide.LEFT: self.left_joint_state_callback,
            ArmSide.RIGHT: self.right_joint_state_callback
        }
        self.joint_state_subs = {
            side: self.create_subscription(
                JointState, 
                MEASURED_JP_TOPIC.format(side=label),
                joint_state_callbacks[side],
                10
            )
            for side, label in arm_labels.items()
        }
        self.joint_state = {ArmSide.LEFT: None, ArmSide.RIGHT: None}

        # Measured poses relative to camera frame
        pose_callbacks = {
            ArmSide.LEFT: self.left_pose_callback,
            ArmSide.RIGHT: self.right_pose_callback
        }
        self.pose_subs = {
            side: self.create_subscription(
                PoseWithCovarianceStamped,
                f'/smoother_uterus/{label}/tip_pose',
                pose_callbacks[side],
                10
            )
            for side, label in arm_labels.items()
        }
        self.measured_poses = {ArmSide.LEFT: None, ArmSide.RIGHT: None}

        # We need the jacobians that relate joint value directions to camera frame positions
        jacobian_callbacks = {
            ArmSide.LEFT: self.left_jacobian_callback,
            ArmSide.RIGHT: self.right_jacobian_callback
        }
        self.jacobian_subs = {
            side: self.create_subscription(
                Float64MultiArray, 
                f'/smoother_uterus/{label}/jac_tip_pose',
                jacobian_callbacks[side],
                10
            )
            for side, label in arm_labels.items()
        }
        self.jacobians = {ArmSide.LEFT: None, ArmSide.RIGHT: None}

        # Cautery control publishers
        self.cautery_pubs = {
            side: self.create_publisher(Float64, SERVO_TOOL_TOPIC.format(side=label), 1)
            for side, label in arm_labels.items()
        }
        self.cautery_mode = 0
        self.cautery_arm = ArmSide.RIGHT
        self.cautery_last_toggle_time = 0.0
        self.cautery_is_on = False

        # We want the action execute callback to run without interupption
        self.server = ActionServer(
            self, 
            VisualServo, 
            '/visual_servoing/visual_servo',
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
    
    def camera_info_callback(self, msg):
        self.fx = msg.p[0]
        self.fy = msg.p[5]
        self.cx = msg.p[2]
        self.cy = msg.p[6] 

        self.get_logger().info("Camera parameters initialized, deleting camera info subscriber.")
        self.destroy_subscription(self.camera_info_sub)  # only run this callback once

    def on_parameter_update(self, params):
        for param in params:
            if param.name == 'cautery_flicker_interval_waypoints':
                self.cautery_flicker_interval_waypoints = int(param.value)
                self.get_logger().info(
                    f"Updated cautery_flicker_interval_waypoints={self.cautery_flicker_interval_waypoints}"
                )
            elif param.name == 'cautery_flicker_off_duration_sec':
                self.cautery_flicker_off_duration_sec = float(param.value)
                self.get_logger().info(
                    f"Updated cautery_flicker_off_duration_sec={self.cautery_flicker_off_duration_sec}"
                )
        return SetParametersResult(successful=True)

    def touch_sensing_callback(self, msg):
        self.is_touching = msg.is_touching

    def robot_cmd_callback(self, msg):
        if len(msg.data) >= 8:
            self.last_robot_cmd = np.array(msg.data[:8], dtype=float)

    def left_joint_state_callback(self, msg):
        self.joint_state[ArmSide.LEFT] = np.array(msg.position[:4], dtype=float)

    def right_joint_state_callback(self, msg):
        self.joint_state[ArmSide.RIGHT] = np.array(msg.position[:4], dtype=float)
    
    def left_pose_callback(self, msg):
        self.measured_poses[ArmSide.LEFT] = msg
    
    def right_pose_callback(self, msg):
        self.measured_poses[ArmSide.RIGHT] = msg
    
    def left_jacobian_callback(self, msg):
        self.jacobians[ArmSide.LEFT] = np.array(msg.data, dtype=float).reshape((6,4))

    def right_jacobian_callback(self, msg):
        self.jacobians[ArmSide.RIGHT] = np.array(msg.data, dtype=float).reshape((6,4))
    
    def tf_waypoint(self, waypoint, from_frame, to_frame):
        # Make a point stamped with no time stamp
        p = PointStamped()
        p.point.x = waypoint.x
        p.point.y = waypoint.y
        p.point.z = waypoint.z
        p.header.stamp = rclpy.time.Time().to_msg()
        p.header.frame_id = from_frame

        # tf to the desired frame
        p_new = self.tf_buffer.transform(p, to_frame, timeout=Duration(seconds=2.0))

        waypoint_new = VisualServoWaypoint()
        waypoint_new.x = p_new.point.x
        waypoint_new.y = p_new.point.y
        waypoint_new.z = p_new.point.z
        waypoint_new.tool_rotation = waypoint.tool_rotation
        waypoint_new.time_seconds = waypoint.time_seconds

        return waypoint_new
    
    def uvz_to_xyz_waypoint(self, uvz):
        xyz = VisualServoWaypoint()
        xyz.time_seconds = uvz.time_seconds
        xyz.tool_rotation = uvz.tool_rotation

        u, v = uvz.x, uvz.y, 
        xyz.x = (u - self.cx) * uvz.z / self.fx
        xyz.y = (v - self.cy) * uvz.z / self.fy
        xyz.z = uvz.z

        return xyz  # camera frame

    def xyz_to_uvz_waypoint(self, xyz):
        # Assume p is in camera frame
        if xyz.z < 1e-6:
            raise ValueError("Cannot project point to camera, Z too small or behind camera.")

        uvz = VisualServoWaypoint()
        uvz.time_seconds = xyz.time_seconds
        uvz.tool_rotation = xyz.tool_rotation

        uvz.x = xyz.x * self.fx / xyz.z + self.cx
        uvz.y = xyz.y * self.fy / xyz.z + self.cy
        uvz.z = xyz.z

        return uvz  # "image" frame: pixels + depth

    def goal_callback(self, goal_request):
        self.get_logger().info("Received new visual servoing goal request. Validating goal...")
        self.get_logger().info(
            f"Goal cautery_mode={int(goal_request.cautery_mode)} "
            f"cautery_arm='{goal_request.cautery_arm}' "
            f"flicker_interval={self.cautery_flicker_interval_waypoints}"
        )

        if self.fx is None:
            self.get_logger().error("REJECT: No camera info received yet. Cannot proceed with visual servoing.")
            return GoalResponse.REJECT
        
        for side in ArmSide:
            traj = goal_request.left_goal_traj if side == ArmSide.LEFT else goal_request.right_goal_traj

            self.get_logger().info(
                f"{arm_labels[side].upper()} ARM:  "
                f"waypoints: {len(traj.waypoints)}, "
                f"frame_id: {traj.frame_id}, "
                f"tip_speed: {traj.tip_speed}, "
                f"tip_speed_z_limit: {traj.tip_speed_z_limit}"
            )

            # Skip validation if this arm doesnt have any waypoints
            if len(traj.waypoints) == 0:
                continue
            
            # Check if user set tip velocity.
            # For now, we enforce that this must be set, but if we want to add support for arbitrary
            # non-constant velocity profiles later, we can allow the client to also set the waypoint
            # times instead. This will be more complicated and maybe not worth it to deal with,
            # unless someone really needs non-constant velocity in their project.
            if not traj.tip_speed > 0.0:
                self.get_logger().error(f"REJECT: {arm_labels[side]} arm: cannot have non-positive tip_speed.")
                return GoalResponse.REJECT

            # If the frame_id is image, then tip_speed_z_limit must also be set
            if traj.frame_id == "image" and not traj.tip_speed_z_limit > 0.0:
                self.get_logger().error(
                    f"REJECT: {arm_labels[side]} arm: When frame_id is 'image', you must set tip_speed_z_limit > 0, rejecting goal."
                )
                return GoalResponse.REJECT

            # Check that we have measured poses, jacobians, and joint states for all arms
            if self.measured_poses[side] is None:
                self.get_logger().error(
                    f"No measured smoother_uterus pose received yet for {arm_labels[side]} arm, rejecting goal."
                )
                return GoalResponse.REJECT

            if self.jacobians[side] is None:
                self.get_logger().error(
                    f"No smoother_uterus jacobian received yet for {arm_labels[side]} arm, rejecting goal."
                )
                return GoalResponse.REJECT
            
            if self.joint_state[side] is None:
                self.get_logger().error(
                    f"No joint state on {MEASURED_JP_TOPIC.format(side=arm_labels[side])} yet, rejecting goal."
                )
                return GoalResponse.REJECT

        # Both arms share one command message, so we need a starting command for the idle arm too
        if self.last_robot_cmd is None and any(self.joint_state[s] is None for s in ArmSide):
            self.get_logger().error(
                f"REJECT: no command seen on {ROBOT_CMD_TOPIC} and no joint state for both arms, "
                f"so the idle arm can't be held in place."
            )
            return GoalResponse.REJECT

        if int(goal_request.cautery_mode) != 0 and self.cautery_pubs[ArmSide.RIGHT].get_subscription_count() == 0:
            self.get_logger().warn(
                f"cautery_mode={int(goal_request.cautery_mode)} requested, but nothing subscribes to "
                f"{SERVO_TOOL_TOPIC}; the robot will move without energizing anything."
            )

        # It would be hard to validate frame_ids here, so we will just trust the client
        self.get_logger().info("Goal accepted.")

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def set_cautery(self, side, on, silent=False):
        msg = Float64(data=1.0 if on else 0.0)
        self.cautery_pubs[side].publish(msg)
        self.cautery_is_on = on
        if not silent:
            state = "on" if on else "off"
            self.get_logger().info(f"Cautery turned {state} on {arm_labels[side]} arm")

    def normalize_cautery_arm(self, arm_str):
        arm_str = (arm_str or "").strip().lower()
        if arm_str == "left":
            return ArmSide.LEFT
        if arm_str == "right":
            return ArmSide.RIGHT
        self.get_logger().warn(f"Unknown cautery_arm '{arm_str}', defaulting to right.")
        return ArmSide.RIGHT

    def insert_start_waypoint(self, side):
        self.update_measured(side)
        m = self.arm_state[side]['measured']

        # Make the start waypoint the current measured position
        start_waypoint = VisualServoWaypoint()
        start_waypoint.x = m.x
        start_waypoint.y = m.y
        start_waypoint.z = m.z
        start_waypoint.tool_rotation = m.tool_rotation
        start_waypoint.time_seconds = 0.0
        
        # Insert the current measured waypoint into the goal list
        self.arm_state[side]['goal_traj'].waypoints.insert(0, start_waypoint)

    def current_arm_cmd(self, side):
        """Current command-space joints [ir, or, it, ot] for one arm."""
        if self.last_robot_cmd is not None:
            return self.last_robot_cmd[ROBOT_CMD_SLICE[side]].copy()

        # Nothing seen on ROBOT_CMD_TOPIC since this node started. Measured == command only if no
        # keyboard offsets are active (true right after /set_home).
        self.get_logger().warn(
            f"No command seen on {ROBOT_CMD_TOPIC} yet; seeding {arm_labels[side]} arm from measured "
            f"joints (wrong if keyboard offsets are active)."
        )
        return self.joint_state[side].copy()

    def init_cmd_joint_state(self, side):
        self.arm_state[side]['cmd_joint_state'] = self.robot_cmd[ROBOT_CMD_SLICE[side]].copy()

    def fill_in_waypoint_times(self, side):
        traj = self.arm_state[side]['goal_traj']
        waypoints = traj.waypoints

        p = np.array([[wp.x, wp.y, wp.z] for wp in waypoints])
        dp = np.diff(p, axis=0)  # shape (N-1, 3)

        if traj.frame_id == 'image':
            # XY timing (pixels/sec)
            pixel_dist = np.linalg.norm(dp[:, :2], axis=1)
            dt_pixels = pixel_dist / traj.tip_speed

            # Z velocity constraint
            z_dist = np.abs(dp[:, 2])
            dt_z = z_dist / traj.tip_speed_z_limit

            # Enforce both constraints
            dt = np.maximum(dt_pixels, dt_z)
        else:
            dist = np.linalg.norm(dp, axis=1)
            dt = dist / traj.tip_speed

        # Fill times
        waypoints[0].time_seconds = 0.0
        for i in range(1, len(waypoints)):
            waypoints[i].time_seconds = waypoints[i - 1].time_seconds + dt[i - 1]

    def init_servo(self, goal_handle):
        self.arm_state = {ArmSide.LEFT: {}, ArmSide.RIGHT: {}}
        self.t_0 = self.get_clock().now()
        self.t = self.get_clock().now()
        self.t_elapsed_sec = 0.0
        self.servo_dt_sec = 0.0
        self.cautery_mode = int(goal_handle.request.cautery_mode)
        self.cautery_arm = self.normalize_cautery_arm(goal_handle.request.cautery_arm)
        self.cautery_last_toggle_time = 0.0
        self.cautery_is_on = False
        self.cautery_flicker_off_until = 0.0
        self.get_logger().info(
            f"Init cautery: mode={self.cautery_mode}, "
            f"interval={self.cautery_flicker_interval_waypoints}, "
            f"off_duration={self.cautery_flicker_off_duration_sec}"
        )

        # Full 8-joint command; arms without waypoints are held at their current command
        self.robot_cmd = np.zeros(8, dtype=float)
        for side in ArmSide:
            self.robot_cmd[ROBOT_CMD_SLICE[side]] = self.current_arm_cmd(side)

        for side in ArmSide:
            goal_traj = goal_handle.request.left_goal_traj if side == ArmSide.LEFT else goal_handle.request.right_goal_traj

            if len(goal_traj.waypoints) == 0:
                self.get_logger().info(f"No waypoints for {arm_labels[side]} arm, skipping servo for this arm.")
                self.arm_state[side]['is_moving'] = False
                continue
            
            self.arm_state[side]['is_moving'] = True
            self.arm_state[side]['segment_idx'] = None
            self.arm_state[side]['last_segment_idx'] = None
            self.arm_state[side]['goal_traj'] = goal_traj
            self.insert_start_waypoint(side)
            self.fill_in_waypoint_times(side)
            self.arm_state[side]['goal_traj_camera_waypoints'] = self.build_camera_waypoints(side)
            self.init_cmd_joint_state(side)
        
        self.result = VisualServo.Result()

    def update_cautery(self):
        if self.cautery_mode == 0:
            return

        # Turn on immediately for ON or FLICKER modes
        if not self.cautery_is_on:
            self.set_cautery(self.cautery_arm, True, silent=True)

        if self.cautery_mode == 1:
            return

        interval = int(self.cautery_flicker_interval_waypoints)
        if interval <= 0:
            # No waypoint-based flicker requested: behave like cautery ON
            return

        # FLICKER mode (waypoint-based): toggle OFF briefly at waypoint intervals
        if not self.arm_state.get(self.cautery_arm, {}).get('is_moving', False):
            return

        segment_idx = self.arm_state[self.cautery_arm].get('segment_idx')
        if segment_idx is None:
            return

        # Turn back on after the off-duration window
        if (not self.cautery_is_on) and (self.t_elapsed_sec >= self.cautery_flicker_off_until):
            self.set_cautery(self.cautery_arm, True, silent=True)

        last_idx = self.arm_state[self.cautery_arm].get('last_segment_idx')
        if last_idx is None:
            self.arm_state[self.cautery_arm]['last_segment_idx'] = segment_idx
            return

        if segment_idx != last_idx:
            self.arm_state[self.cautery_arm]['last_segment_idx'] = segment_idx
            self.get_logger().info(
                f"Cautery segment idx={segment_idx}, interval={interval}, "
                f"mode={self.cautery_mode}"
            )
            if (segment_idx + 1) % interval == 0:
                self.set_cautery(self.cautery_arm, False, silent=True)
                self.cautery_flicker_off_until = self.t_elapsed_sec + self.cautery_flicker_off_duration_sec

    def get_interp_waypoints(self, side):
        waypoints = self.arm_state[side]['goal_traj'].waypoints

        # Search the intervals between the waypoints (includes inserted start waypoint)
        for i in range(len(waypoints) - 1):
            t_i = waypoints[i].time_seconds
            t_ip1 = waypoints[i + 1].time_seconds
            if t_i <= self.t_elapsed_sec < t_ip1:
                return (waypoints[i], waypoints[i + 1], i)

        # Else, its greater than final waypoint: return two final waypoints
        # This ultimiately sets the interpolated value to a constant
        last_idx = max(len(waypoints) - 2, 0)
        return (waypoints[-1], waypoints[-1], last_idx)

    def build_camera_waypoints(self, side):
        waypoints = self.arm_state[side]['goal_traj'].waypoints
        if self.arm_state[side]['goal_traj'].frame_id == 'image':
            return [self.uvz_to_xyz_waypoint(wp) for wp in waypoints]
        return [
            self.tf_waypoint(wp, self.arm_state[side]['goal_traj'].frame_id, 'smoother_uterus/camera')
            for wp in waypoints
        ]

    def update_setpoint(self, side):
        # First get the two interpolation waypoints for the current time
        wp = self.get_interp_waypoints(side)
        seg_idx = wp[2]
        
        # Now linear interpolate between them
        dt = wp[1].time_seconds - wp[0].time_seconds
        alpha = 0.0 if dt < 1e-9 else (self.t_elapsed_sec - wp[0].time_seconds) / dt

        s = VisualServoWaypoint()
        s.time_seconds = self.t_elapsed_sec
        s.x = wp[0].x * (1 - alpha) + wp[1].x * alpha
        s.y = wp[0].y * (1 - alpha) + wp[1].y * alpha
        s.z = wp[0].z * (1 - alpha) + wp[1].z * alpha
        s.tool_rotation = wp[0].tool_rotation * (1 - alpha) + wp[1].tool_rotation * alpha

        wp_camera = self.arm_state[side]['goal_traj_camera_waypoints']
        wp0_camera = wp_camera[seg_idx]
        wp1_camera = wp_camera[seg_idx + 1]
        s_camera = VisualServoWaypoint()
        s_camera.time_seconds = self.t_elapsed_sec
        s_camera.x = wp0_camera.x * (1 - alpha) + wp1_camera.x * alpha
        s_camera.y = wp0_camera.y * (1 - alpha) + wp1_camera.y * alpha
        s_camera.z = wp0_camera.z * (1 - alpha) + wp1_camera.z * alpha
        s_camera.tool_rotation = wp0_camera.tool_rotation * (1 - alpha) + wp1_camera.tool_rotation * alpha

        self.arm_state[side]['setpoint'] = s
        self.arm_state[side]['setpoint_camera'] = s_camera
        self.arm_state[side]['segment_idx'] = seg_idx

        # Feedforward tip velocity in camera frame from the current waypoint segment
        if dt < 1e-9:
            p_dot_ff = np.zeros(3, dtype=float)
        else:
            p_dot_ff = np.array(
                [wp1_camera.x - wp0_camera.x, wp1_camera.y - wp0_camera.y, wp1_camera.z - wp0_camera.z],
                dtype=float
            ) / dt

        self.arm_state[side]['setpoint_camera_dot'] = p_dot_ff
    
    def update_measured(self, side):
        m_camera = VisualServoWaypoint()
        m_camera.time_seconds = self.t_elapsed_sec

        p = self.measured_poses[side].pose.pose.position
        m_camera.x = p.x
        m_camera.y = p.y
        m_camera.z = p.z
        # Inner rotation can't be measured by the smoother, so report the current COMMAND (same space as
        # the waypoints). The measured joint includes keyboard offsets (e.g. the screw sequence's turns),
        # and starting the interpolation from it would spin the tool back by that offset.
        m_camera.tool_rotation = float(self.robot_cmd[ROBOT_CMD_SLICE[side]][0])

        if self.arm_state[side]['goal_traj'].frame_id == 'image':
            m = self.xyz_to_uvz_waypoint(m_camera)
        else:
            m = self.tf_waypoint(m_camera, 'smoother_uterus/camera', self.arm_state[side]['goal_traj'].frame_id)

        self.arm_state[side]['measured_camera'] = m_camera
        self.arm_state[side]['measured'] = m
    
    def update_cmd_joint_state(self, side):
        p_meas_msg = self.arm_state[side]['measured_camera']
        p_set_msg  = self.arm_state[side]['setpoint_camera']
        p_meas = np.array([p_meas_msg.x, p_meas_msg.y, p_meas_msg.z], dtype=float)
        p_set  = np.array([p_set_msg.x,  p_set_msg.y,  p_set_msg.z],  dtype=float)
        tool_rotation_set = self.arm_state[side]['setpoint_camera'].tool_rotation

        # We only use the bottom 3 rows for linear velocity control
        J = self.jacobians[side][3:].copy()
        
        Kp = self.vs_kp
        Kff = self.vs_kff
        p_dot_ff = self.arm_state[side].get('setpoint_camera_dot', np.zeros(3, dtype=float))
        p_error = p_set - p_meas

        dt = max(self.servo_dt_sec, 1e-6)
        p_dot = Kff * p_dot_ff + Kp * p_error
        q_dot = np.linalg.pinv(J) @ p_dot
        q_dot = clip_joint_speed(q_dot)

        q_prev = self.arm_state[side]['cmd_joint_state']
        q = q_prev + q_dot * dt
        q = clip_joint_values(q)

        # Set tool rotation directly since we can't measure it.
        # Note, this joint does not affect tip position, only tool roll.
        q[0] = tool_rotation_set

        self.arm_state[side]['cmd_joint_state'] = q

    def send_robot_cmd(self):
        """Publish both arms in one message: moving arms from their servo command, idle arms held."""
        for side in ArmSide:
            if self.arm_state[side]['is_moving']:
                self.robot_cmd[ROBOT_CMD_SLICE[side]] = self.arm_state[side]['cmd_joint_state']

        cmd = Float32MultiArray()
        cmd.data = [float(v) for v in self.robot_cmd]
        self.robot_cmd_pub.publish(cmd)
    
    def update_single_servo(self):
        self.t = self.get_clock().now()
        t_elapsed_sec = (self.t - self.t_0).nanoseconds / 1.0e9
        self.servo_dt_sec = max(t_elapsed_sec - self.t_elapsed_sec, 1e-6)
        self.t_elapsed_sec = t_elapsed_sec

        for side in ArmSide:
            if not self.arm_state[side]['is_moving']:
                continue
            
            t0 = time.monotonic()
            self.update_setpoint(side)

            t1 = time.monotonic()
            self.update_measured(side)

            t2 = time.monotonic()
            self.update_cmd_joint_state(side)

            t3 = time.monotonic()
            elapsed = t3 - t0
            setpoint = t1 - t0
            measured = t2 - t1
            update_cmd = t3 - t2

            # If there is a time problem, print the breakdown of where time is spent
            if elapsed > 0.01:
                self.get_logger().warn(
                    f"  {arm_labels[side].upper()} ARM timing breakdown (total {elapsed*1e3:.1f} ms): "
                    f"setpoint={setpoint*1e3:.1f} ms, "
                    f"measured={measured*1e3:.1f} ms, "
                    f"update_cmd={update_cmd*1e3:.1f} ms"
                )

        # One message carries both arms on the uterine robot
        self.send_robot_cmd()

    def store_result_get_feedback(self):
        feedback = VisualServo.Feedback()

        for side in ArmSide:
            if not self.arm_state[side]['is_moving']:
                continue

            m, s = self.arm_state[side]['measured'], self.arm_state[side]['setpoint']

            if side == ArmSide.LEFT:
                self.result.left_measured_traj.append(m)
                self.result.left_setpoint_traj.append(s)
                feedback.left_measured = m
                feedback.left_setpoint = s
            else:
                self.result.right_measured_traj.append(m)
                self.result.right_setpoint_traj.append(s)
                feedback.right_measured = m
                feedback.right_setpoint = s
        
        return feedback

    def check_stop_conditions(self, goal_handle):
        # Stop if cancel requested
        if goal_handle.is_cancel_requested:
            self.get_logger().info('Received cancel command, canceling visual servo action.')
            goal_handle.canceled()
            self.result.success = False
            self.result.stopped_touch = False
            return True
        
        # Stop if touch detected and that behavior was requested
        if self.is_touching and goal_handle.request.stop_on_touch:
            self.get_logger().warn(f"Proprioception detected touch point, aborting servo action.")
            goal_handle.abort()
            self.result.success = False
            self.result.stopped_touch = True
            return True
        
        # If we are over the final time, then stop moving
        both_done = True  # assume done until we find an arm that is still moving
        for side in ArmSide:
            if self.arm_state[side]['is_moving']:
                last_time = self.arm_state[side]['goal_traj'].waypoints[-1].time_seconds
                if self.t_elapsed_sec < last_time:
                    both_done = False
                    break

        if both_done:
            self.get_logger().info("Time exceeded last waypoint times, trajectory executed successfully.")
            goal_handle.succeed()
            self.result.success = True
            self.result.stopped_touch = False
            return True
        
        return False
    
    def execute_callback(self, goal_handle):
        self.get_logger().info("Executing trajectories...")
        self.init_servo(goal_handle)

        time_window_sec = 1.0
        recent_update_times = []
        last_log_time = 0.0  # or self.t_elapsed_sec after first update

        rate_hz = 50.0
        rate = self.create_rate(rate_hz, self.get_clock())

        while rclpy.ok():
            self.update_single_servo()
            self.update_cautery()
            recent_update_times.append(self.t_elapsed_sec)

            # Keep only updates within the last time_window_sec
            recent_update_times = [t for t in recent_update_times if (self.t_elapsed_sec - t) <= time_window_sec]

            # Log once per time window
            if self.t_elapsed_sec - last_log_time >= time_window_sec:
                if len(recent_update_times) >= 2:
                    update_rate = len(recent_update_times) / (recent_update_times[-1] - recent_update_times[0])
                    self.get_logger().info(f"Visual servoing update rate: {update_rate:.2f} Hz")
                    last_log_time = self.t_elapsed_sec

            feedback = self.store_result_get_feedback()
            goal_handle.publish_feedback(feedback)

            if self.check_stop_conditions(goal_handle):
                if self.cautery_mode != 0:
                    self.set_cautery(self.cautery_arm, False, silent=True)
                return self.result
            
            rate.sleep()


def main():
    rclpy.init()

    # Need multithreaded executor so that callbacks are processed during servoing loop,
    # otherwise, e.g. measured pose/jacobian updates would not be received -> no servoing
    exec = MultiThreadedExecutor(num_threads=8)  
    node = VisualServoServer()
    exec.add_node(node)

    try:
        exec.spin()
    finally:
        exec.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()