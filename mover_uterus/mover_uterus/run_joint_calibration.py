#!/usr/bin/env python3
"""
Calibration trajectory in JOINT space, for bringing up a new robot.

run_calibration.py moves the tips through an image-space raster with the visual servo. Before
calibration is locked in, the servo steers on the smoother's untuned geometry, so on a new robot the
tips can end up out of view. This script instead sweeps each arm through joint ranges you have
checked by eye keep the tip visible. It needs no geometry at all: the calibrator only needs pairs of
joint values and detected tips, spread over the workspace.

Per arm: wait for /mover_uterus/continue (or task "calibrate"), move slowly from the current pose to
the first waypoint, sweep a serpentine grid over (outer translation, outer rotation, inner extension)
while rolling the inner tube, then return slowly to home. The other arm is held where it is.
After both arms it sends /smoother_uterus/stop_calibration to lock the calibration in.

Commands go straight to motor_node on /robot/state/current_state. Don't run servo goals, the IK
node or keyboard control at the same time. Try it with sim_motor_node.py first.

Run:  ros2 run mover_uterus run_joint_calibration
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray, String

# ---------------------------------------------------------------------------------------------------
# Joint ranges where the tip stays in view. Find them by jogging one arm and watching the image, e.g.
#   ros2 topic pub -1 /robot/state/current_state std_msgs/msg/Float32MultiArray \
#     "{data: [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.03, 0.015]}"      # left: inner 30 mm, outer 15 mm
# Per-joint layout of each arm: [inner_rot, outer_rot, inner_trans, outer_trans] (rad, m)
# ---------------------------------------------------------------------------------------------------
RANGES = {
    'left': {
        'outer_trans': (0.010, 0.025),          # m; below the minimum the tube hasn't curved down into view
        'inner_extra': (0.005, 0.025),          # m the inner tube sticks out past the outer
        'outer_rot': (np.radians(-30), np.radians(30)),  # 0 bends the tubes down toward the camera
    },
    'right': {
        'outer_trans': (0.010, 0.03),
        'inner_extra': (0.005, 0.03),
        'outer_rot': (np.radians(-10), np.radians(50)),
    },
}
N_OUTER_TRANS = 5      # levels of outer translation
N_OUTER_ROT = 5        # levels of outer rotation (swept back and forth)
N_INNER = 3            # levels of inner extension
ROLL_PATTERN = [0.0, np.pi, 0.0, -np.pi]   # inner tube roll, cycled per waypoint (helps convergence)

# Motion
TRANS_SPEED = 0.003            # m/s, slowest-joint pacing between waypoints
ROT_SPEED = np.radians(20)     # rad/s, outer rotation (swings the tip)
ROLL_SPEED = np.radians(60)    # rad/s, inner rotation (rolls the tool in place)
DWELL_S = 1.0                  # pause at each waypoint (sharper images, more samples)
RATE_HZ = 50.0

# Hard stops, same as the servo
INNER_MAX, OUTER_MAX = 0.06, 0.04

ROBOT_CMD_TOPIC = '/robot/state/current_state'
ARM_SLICE = {'right': slice(0, 4), 'left': slice(4, 8)}   # [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]


def arm_waypoints(r):
    """Serpentine grid over (outer_trans, outer_rot, inner_extra), neighbours one step apart."""
    ots = np.linspace(*r['outer_trans'], N_OUTER_TRANS)
    rots = np.linspace(*r['outer_rot'], N_OUTER_ROT)
    extras = np.linspace(*r['inner_extra'], N_INNER)
    points = []
    for i, ot in enumerate(ots):
        rot_order = rots if i % 2 == 0 else rots[::-1]
        for j, rot in enumerate(rot_order):
            extra_order = extras if (i * N_OUTER_ROT + j) % 2 == 0 else extras[::-1]
            for extra in extra_order:
                roll = ROLL_PATTERN[len(points) % len(ROLL_PATTERN)]
                points.append(np.array([roll, rot, ot + extra, ot]))
    return points


def clip_arm(q):
    q = q.copy()
    q[3] = np.clip(q[3], 0.0, OUTER_MAX)
    q[2] = np.clip(q[2], q[3], INNER_MAX)   # inner never behind the outer
    return q


class JointCalibrationRunner(Node):
    def __init__(self):
        super().__init__('joint_calibration_runner')
        self.cmd_pub = self.create_publisher(Float32MultiArray, ROBOT_CMD_TOPIC, 1)
        self.stop_calibration_pub = self.create_publisher(Empty, '/smoother_uterus/stop_calibration', 10)

        self.last_cmd = None
        self.create_subscription(Float32MultiArray, ROBOT_CMD_TOPIC, self.cmd_callback, 10)
        self.measured = {'left': None, 'right': None}
        for arm in ('left', 'right'):
            self.create_subscription(
                JointState, f'/robot/{arm}/joint/measured_jp',
                lambda msg, arm=arm: self.measured.__setitem__(arm, np.array(msg.position[:4], dtype=float)), 10)

    def cmd_callback(self, msg):
        if len(msg.data) >= 8:
            self.last_cmd = np.array(msg.data[:8], dtype=float)

    def have_measured(self):
        # `None in dict.values()` would compare numpy arrays to None elementwise, so check identity
        return all(v is not None for v in self.measured.values())

    def current_command(self):
        """Full 8-joint command to start from: last command seen, else measured joints (no offsets)."""
        deadline = time.time() + 3.0
        while rclpy.ok() and time.time() < deadline and self.last_cmd is None and not self.have_measured():
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.last_cmd is not None:
            return self.last_cmd.copy()
        if self.have_measured():
            self.get_logger().warn('No command seen yet; starting from measured joints (assumes no keyboard offsets)')
            q = np.zeros(8)
            for arm, sl in ARM_SLICE.items():
                q[sl] = self.measured[arm]
            return q
        raise RuntimeError(f'No command on {ROBOT_CMD_TOPIC} and no joint states: is motor_node running?')

    def publish(self, q):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in q]
        self.cmd_pub.publish(msg)

    def move_arm(self, q_full, arm, target):
        """Linear move of one arm to target at the joint speed limits; returns the new full command."""
        sl = ARM_SLICE[arm]
        start = q_full[sl].copy()
        target = clip_arm(target)
        delta = np.abs(target - start)
        duration = max(delta[0] / ROLL_SPEED, delta[1] / ROT_SPEED,
                       delta[2] / TRANS_SPEED, delta[3] / TRANS_SPEED, 0.2)
        steps = max(1, int(duration * RATE_HZ))
        for k in range(1, steps + 1):
            if not rclpy.ok():
                break
            q_full[sl] = start + (target - start) * k / steps
            self.publish(q_full)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / RATE_HZ)
        return q_full

    def dwell(self, q_full, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            self.publish(q_full)   # keep the command fresh
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / RATE_HZ)

    def wait_for_continue(self, arm):
        self.get_logger().info(
            f'{arm} arm next: publish /mover_uterus/continue or type "calibrate" in task_publisher')
        self.continue_received = False
        sub = self.create_subscription(Empty, '/mover_uterus/continue',
                                       lambda _: setattr(self, 'continue_received', True), 10)
        task_sub = self.create_subscription(
            String, '/current_task',
            lambda msg: setattr(self, 'continue_received', self.continue_received or msg.data.strip() == 'calibrate'),
            10)
        while rclpy.ok() and not self.continue_received:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.destroy_subscription(sub)
        self.destroy_subscription(task_sub)

    def run_arm(self, arm):
        self.wait_for_continue(arm)
        q = self.current_command()
        waypoints = arm_waypoints(RANGES[arm])
        self.get_logger().info(f'{arm} arm: {len(waypoints)} waypoints')
        for i, wp in enumerate(waypoints):
            ir, orr, it, ot = clip_arm(wp)
            self.get_logger().info(
                f'{arm} {i + 1}/{len(waypoints)}: outer {ot * 1000:.1f} mm, inner {it * 1000:.1f} mm, '
                f'outer rot {np.degrees(orr):.0f} deg, roll {np.degrees(ir):.0f} deg')
            q = self.move_arm(q, arm, wp)
            self.dwell(q, DWELL_S)

        # Back home the same careful way: retract the inner tube to the outer, then both to zero
        sl = ARM_SLICE[arm]
        cur = q[sl].copy()
        q = self.move_arm(q, arm, np.array([cur[0], cur[1], cur[3], cur[3]]))
        q = self.move_arm(q, arm, np.array([0.0, 0.0, 0.0, 0.0]))
        self.get_logger().info(f'{arm} arm done, at home')

    def run(self):
        self.run_arm('left')
        self.run_arm('right')
        self.get_logger().info('Sending /smoother_uterus/stop_calibration to lock in the calibration')
        self.stop_calibration_pub.publish(Empty())
        self.dwell(self.current_command(), 1.0)


def main():
    rclpy.init()
    node = JointCalibrationRunner()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn('Interrupted: the robot holds the last command sent')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()