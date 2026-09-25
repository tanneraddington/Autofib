#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, Int32
from geometry_msgs.msg import Twist


class StateMux(Node):
    def __init__(self):
        super().__init__('state_mux')

        # --- State ---
        self.override_active = False
        self.last_constant_msg = None

        # --- Subscribers ---
        self.create_subscription(
            Float32MultiArray,
            '/robot/state/current_state_r_in',
            self.constant_cb,
            10
        )

        self.create_subscription(
            Int32,
            '/set_home',
            self.set_home_cb,
            10
        )

        self.create_subscription(
            Twist,
            '/ui_twist_r',
            self.twist_cb,
            10
        )

        # --- Publisher ---
        self.pub = self.create_publisher(
            Float32MultiArray,
            '/robot/state/current_state_r',
            10
        )

        # --- Timer for override publishing ---
        self.timer = self.create_timer(0.02, self.publish_override)

        self.get_logger().info('State mux node started')

    # ---------------- Callbacks ----------------

    def constant_cb(self, msg: Float32MultiArray):
        self.last_constant_msg = msg
        if not self.override_active:
            self.pub.publish(msg)

    def set_home_cb(self, msg: Int32):
        if msg.data == 1:
            if not self.override_active:
                self.get_logger().info('Set-home received: activating override')
            self.override_active = True

    def twist_cb(self, msg: Twist):
        if not self.override_active:
            return

        if self.release_condition_met(msg):
            self.get_logger().info(
                'Release condition met. Returning control.'
            )
            self.override_active = False

            # Immediately restore last known constant state
            if self.last_constant_msg is not None:
                self.pub.publish(self.last_constant_msg)

    # ---------------- Helpers ----------------

    def publish_override(self):
        if not self.override_active:
            return

        msg = Float32MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.pub.publish(msg)

    def release_condition_met(self, msg: Twist) -> bool:
        lin_x_ok = abs(msg.linear.x) <= 0.1
        lin_y_ok = abs(msg.linear.y) <= 0.1
        ang_z_ok = abs(msg.angular.z + 1.0) <= 0.1  # close to -1.0

        return lin_x_ok and lin_y_ok and ang_z_ok


def main(args=None):
    rclpy.init(args=args)
    node = StateMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
