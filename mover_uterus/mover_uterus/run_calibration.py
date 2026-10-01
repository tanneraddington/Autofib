import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

import numpy as np

from mover_uterus.action_client import SmootherMover


def get_raster_pattern_pixels(num_rows, theta, width=711, height=400, x_margin=0, y_margin=200):
    x_min = x_margin
    x_max = width - x_margin
    y_min = y_margin
    y_max = height - y_margin

    x = []
    for row in range(num_rows):
        if row % 2 == 0:
            x.extend([x_min, x_max])
        else:
            x.extend([x_max, x_min])

    y_rows = np.linspace(y_min, y_max, num_rows)
    y = np.repeat(y_rows, 2)

    pixels = np.column_stack((x, y))

    # Apply rotation around center of image
    center = np.array([width / 2, height / 2])
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    rotated_pixels = []
    for pixel in pixels:
        centered = pixel - center
        rotated = rotation_matrix @ centered
        final = rotated + center
        rotated_pixels.append(final)
    
    return np.array(rotated_pixels)


class CalibrationTrajectoryRunner(Node):
    def __init__(self):
        super().__init__('calibration_trajectory_runner')

        self.stop_calibration_pub = self.create_publisher(
            Empty, '/smoother_uterus/stop_calibration', 10
        )
        self.mover = SmootherMover()

    def run_arm_trajectory(self, arm, z_min=0.015, z_max=0.03, num_raster_rows=5, num_depth_levels=3):
        
        # Wait for continue signal before starting the motion 
        self.get_logger().info(
            "Waiting for /mover_uterus/continue (or task 'calibrate' on /current_task) to proceed...")
        self.wait_for_continue()
        self.get_logger().info("Received signal. Starting trajectory for arm: " + arm)

        # Raster tilt per arm.
        if arm == "left":
            z_offset = 0.0
            raster_angle = np.radians(18)
        else:
            z_offset = 0.004
            raster_angle = np.radians(-18)


        self.mover.go_home()

        z_min += z_offset
        z_max += z_offset

        p = [[0.0, 0.0, 0.02 + z_offset]]
        self.mover.servo_arm(arm, p, "smoother_uterus/camera")

        # Start far, end close
        z_values = np.linspace(z_max, z_min, num_depth_levels)
        pixels = get_raster_pattern_pixels(num_rows=num_raster_rows, theta=raster_angle)
        
        # Do serpentine pattern through pixels at each depth value
        points = []
        for i, z in enumerate(z_values):
            # Do reverse order of pixels every other depth level s.t. we don't go back to top left every time
            pix = pixels if i % 2 == 0 else pixels[::-1]
            for pixel in pix:
                points.append(np.array([pixel[0], pixel[1], z]))

        # Funny roll pattern to not always be moving in the same direction, which seems to help with calibration convergence
        roll_pattern = np.array([0.0, np.pi, 0.0, -np.pi, 0.0, -np.pi, 0.0, np.pi])
        rots = np.tile(roll_pattern, len(points) // len(roll_pattern) + 1)[:len(points)]

        self.mover.servo_arm(
            arm,
            points,
            "image",
            tip_speed=200.0,
            tool_rotation=rots.tolist()
        )

        self.mover.go_home()

    def wait_for_continue(self):
        self.continue_received = False
        sub = self.create_subscription(
            Empty, '/mover_uterus/continue', lambda _: setattr(self, 'continue_received', True), 10
        )
        # Also accept the uterine task dispatcher (task_publisher_node.py): type "calibrate"
        task_sub = self.create_subscription(
            String, '/current_task',
            lambda msg: setattr(self, 'continue_received', self.continue_received or msg.data.strip() == 'calibrate'),
            10
        )
        
        # Wait until someone publishes the signal 
        while not self.continue_received:
            rclpy.spin_once(self, timeout_sec=0.5)

        self.destroy_subscription(sub)
        self.destroy_subscription(task_sub)

    def run_trajectory(self):
        self.run_arm_trajectory("left")
        self.run_arm_trajectory("right")

        self.mover.destroy_node()

        self.get_logger().info("Sending stop signal to smoother_uterus.")
        self.stop_calibration_pub.publish(Empty())


def main():
    rclpy.init()
    
    runner = CalibrationTrajectoryRunner()
    try:
        runner.run_trajectory()
    finally:
        runner.destroy_node()
        rclpy.shutdown()