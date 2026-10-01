#!/usr/bin/env python3
"""
Stand-in for motor_node, for dry runs: takes the same 8-joint command on /robot/state/current_state and
publishes /robot/<side>/joint/measured_jp as if the robot had moved there (speed-limited like the
Dynamixels). The smoother, servo and mover run unchanged, so you can watch a calibration or test trajectory
in rviz and read every joint command before the real robot moves.

DO NOT run this at the same time as motor_node: both listen to the same command topic.

Run:  python3 sim_motor_node.py      (Ctrl-C prints the range each joint was commanded over)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

ROBOT_CMD_TOPIC = '/robot/state/current_state'
RATE_HZ = 30.0

# Approximate Dynamixel profile speeds from motor_node.cpp (profile velocity ~176-200 x 0.229 rpm)
MAX_TRANS_SPEED = 0.013   # m/s   (~40 rpm on a 20 mm lead screw)
MAX_ROT_SPEED = 4.8       # rad/s (~46 rpm)

# Warnings (edit to what is safe on your robot)
INNER_MAX, OUTER_MAX = 0.06, 0.04     # m
JUMP_TRANS, JUMP_ROT = 0.003, 0.3     # m, rad between consecutive commands

NAMES = ['inner_rotation', 'outer_rotation', 'inner_translation', 'outer_translation']
ARMS = {'right': slice(0, 4), 'left': slice(4, 8)}   # command layout: [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]
IS_TRANS = np.array([False, False, True, True] * 2)


class SimMotorNode(Node):
    def __init__(self):
        super().__init__('sim_motor_node')
        self.q = np.zeros(8)        # simulated joint positions
        self.cmd = np.zeros(8)      # last command
        self.cmd_min = np.zeros(8)
        self.cmd_max = np.zeros(8)
        self.n_cmds = 0

        self.pubs = {arm: self.create_publisher(JointState, f'/robot/{arm}/joint/measured_jp', 10) for arm in ARMS}
        self.create_subscription(Float32MultiArray, ROBOT_CMD_TOPIC, self.cmd_callback, 10)
        self.create_timer(1.0 / RATE_HZ, self.step)
        self.get_logger().info('Simulated robot at home (all joints 0). Real motors are NOT driven.')

    def cmd_callback(self, msg):
        if len(msg.data) < 8:
            self.get_logger().error(f'Command has {len(msg.data)} values, motor_node needs 8')
            return
        new = np.array(msg.data[:8], dtype=float)
        jump = np.abs(new - self.cmd)
        big = np.where(IS_TRANS, jump > JUMP_TRANS, jump > JUMP_ROT)
        if self.n_cmds and big.any():
            for i in np.flatnonzero(big):
                arm = 'right' if i < 4 else 'left'
                self.get_logger().warn(f'Jump in {arm} {NAMES[i % 4]}: {self.cmd[i]:.4f} -> {new[i]:.4f}')
        for arm, sl in ARMS.items():
            ir, orr, it, ot = new[sl]
            if it > INNER_MAX or ot > OUTER_MAX or it < 0 or ot < 0 or ot > it + 1e-6:
                self.get_logger().warn(
                    f'{arm} extension out of range: inner {it * 1000:.1f} mm, outer {ot * 1000:.1f} mm')
        self.cmd = new
        self.cmd_min = np.minimum(self.cmd_min, new)
        self.cmd_max = np.maximum(self.cmd_max, new)
        self.n_cmds += 1

    def step(self):
        dt = 1.0 / RATE_HZ
        limit = np.where(IS_TRANS, MAX_TRANS_SPEED, MAX_ROT_SPEED) * dt
        self.q += np.clip(self.cmd - self.q, -limit, limit)
        stamp = self.get_clock().now().to_msg()
        for arm, sl in ARMS.items():
            js = JointState()
            js.header.stamp = stamp
            js.name = NAMES
            js.position = [float(v) for v in self.q[sl]]
            self.pubs[arm].publish(js)

    def report(self):
        print(f'\n{self.n_cmds} commands received. Commanded range per joint:')
        for arm, sl in ARMS.items():
            for name, lo, hi, trans in zip(NAMES, self.cmd_min[sl], self.cmd_max[sl], IS_TRANS[sl]):
                unit, k = ('mm', 1000) if trans else ('deg', 180 / np.pi)
                print(f'  {arm:5s} {name:18s} {lo * k:8.1f} to {hi * k:8.1f} {unit}')


def main():
    rclpy.init()
    node = SimMotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
