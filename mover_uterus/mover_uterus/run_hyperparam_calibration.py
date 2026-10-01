import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

# Uterine robot (motor_node.cpp): both arms are commanded in one message
#   [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]
ROBOT_CMD_TOPIC = '/robot/state/current_state'
LEFT_SLICE = slice(4, 8)


class HyperParamRunner(Node):
    def __init__(self):
        super().__init__('hyperparam_runner')

        self.robot_cmd_pub = self.create_publisher(Float32MultiArray, ROBOT_CMD_TOPIC, 1)

        # Last full command, so the right arm is held wherever it was commanded
        self.robot_cmd = None
        self.robot_cmd_sub = self.create_subscription(
            Float32MultiArray, ROBOT_CMD_TOPIC, self.robot_cmd_callback, 10)

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/robot/left/joint/measured_jp',
            self.joint_state_callback,
            10
        )

        self.joint_state = None

    def robot_cmd_callback(self, msg):
        if self.robot_cmd is None and len(msg.data) >= 8:
            self.robot_cmd = np.array(msg.data[:8], dtype=float)

    def joint_state_callback(self, msg):
        self.joint_state = np.array(msg.position[:4], dtype=float)

    def send_cmd_joint_state(self, q):
        self.robot_cmd[LEFT_SLICE] = q
        cmd = Float32MultiArray()
        cmd.data = [float(v) for v in self.robot_cmd]
        self.robot_cmd_pub.publish(cmd)

    def sample_waypoint(self, q_prev):
        # outer rot stays away from the direction that bends the tubes into the camera.
        # On the uterine robot outer_rot = 0 bends TOWARD the camera (ARM_BASE_YAW = 0 in SmootherSolver.cpp),
        # so sample a v shape centered on pi instead of 0.
        outer_rot = np.random.uniform(np.pi / 3, 5 / 3 * np.pi)

        # outer pos has full range of actuation (motor_node limit 40 mm)
        outer_pos = np.random.uniform(0.0, 0.04)

        # inner rot can be anything, but only changes by at most 30 degrees from previous to avoid large jumps
        inner_rot = q_prev[0] + np.random.uniform(-np.radians(30), np.radians(30))

        # Ensure inner position is always greater than outer, and that it is at least 0.015 to avoid checkerboard/robot collision
        inner_pos_low = max(outer_pos, 0.015)
        inner_pos = np.random.uniform(inner_pos_low, 0.06)  # 60 mm tools

        return np.array([inner_rot, outer_rot, inner_pos, outer_pos], dtype=float)
    
    def run(self, num_waypoints=1000, time_per_waypoint=15.0):
        # Wait until we have received at least one joint state and one full robot command
        while rclpy.ok() and (self.joint_state is None or self.robot_cmd is None):
            self.get_logger().info(f"Waiting for left joint state and a command on {ROBOT_CMD_TOPIC}...")
            rclpy.spin_once(self, timeout_sec=1.0)
        
        # Start from the current left-arm COMMAND (measured joints include keyboard offsets)
        q_prev = self.robot_cmd[LEFT_SLICE].copy()

        for i in range(num_waypoints):
            self.get_logger().info(f"Moving to waypoint {i+1}/{num_waypoints}")

            q_next = self.sample_waypoint(q_prev)
            t_start = self.get_clock().now()

            while rclpy.ok(): 
                elapsed = self.get_clock().now() - t_start
                dt = elapsed.nanoseconds * 1e-9
                if dt > time_per_waypoint:
                    break

                # Interpolate cmd position between q_prev and q_next
                alpha = dt / time_per_waypoint
                q_cmd = (1 - alpha) * q_prev + alpha * q_next
                self.send_cmd_joint_state(q_cmd)

                time.sleep(0.01)

            q_prev = q_next

        self.get_logger().info("Completed all waypoints.")


def main():
    rclpy.init()
    node = HyperParamRunner()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
