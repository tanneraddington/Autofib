import argparse
import numpy as np
import matplotlib.pyplot as plt

import rclpy
  
from mover_uterus.action_client import SmootherMover


def plot_result(result):
    fig, axs = plt.subplots(4, 2, figsize=(12, 8), sharex='all', sharey='row')

    for arm_idx, arm_name in enumerate(['Left Arm', 'Right Arm']):
        setpoints = result.left_setpoint_traj if arm_idx == 0 else result.right_setpoint_traj
        measured  = result.left_measured_traj  if arm_idx == 0 else result.right_measured_traj

        if len(setpoints) == 0 or len(measured) == 0:
            axs[0, arm_idx].set_title(f'{arm_name} (no data)')
            continue

        ts = np.array([p.time_seconds for p in setpoints], dtype=float)
        tm = np.array([p.time_seconds for p in measured],  dtype=float)

        ps = np.array([[p.x, p.y, p.z] for p in setpoints], dtype=float)
        pm = np.array([[p.x, p.y, p.z] for p in measured],  dtype=float)

        axs[0, arm_idx].set_title(f'{arm_name} Trajectory')
        axs[0, 0].set_ylabel('X Position (m) or (pixels)')
        axs[1, 0].set_ylabel('Y Position (m) or (pixels)')
        axs[2, 0].set_ylabel('Z Position (m) or (pixels)')
        axs[3, 0].set_ylabel('Position error (m) or (pixels)')
        axs[3, 0].set_xlabel('Time (s)')
        axs[3, 1].set_xlabel('Time (s)')

        axs[0, arm_idx].plot(ts, ps[:, 0], label='Setpoint X')
        axs[0, arm_idx].plot(tm, pm[:, 0], label='Measured X', linestyle='--')
        axs[0, arm_idx].legend()

        axs[1, arm_idx].plot(ts, ps[:, 1])
        axs[1, arm_idx].plot(tm, pm[:, 1], linestyle='--')

        axs[2, arm_idx].plot(ts, ps[:, 2])
        axs[2, arm_idx].plot(tm, pm[:, 2], linestyle='--')

        errors = np.linalg.norm(ps - pm, axis=1)
        axs[3, arm_idx].plot(ts, errors)

        total_time = ts[-1] - ts[0]
        num_points = len(ts)
        servo_rate_hz = num_points / total_time if total_time > 0 else 0.0
        print(f"{arm_name} servo rate: {servo_rate_hz:.2f} Hz over {total_time:.2f} s with {num_points} points")

    plt.tight_layout()
    plt.show()


def get_square_waypoints(x_center, y_center, d, z):
    x_min = x_center - d
    x_max = x_center + d
    y_min = y_center - d
    y_max = y_center + d
    
    p = [
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z],
        [x_min, y_min, z],
    ]

    r = [0.0, np.pi/2, 0.0, -np.pi, 0.0]

    return p, r


def run(args):
    rclpy.init()

    mover = SmootherMover()

    if args.mode == "single":
        if args.frame == "image":
            x_center = 355.5
            y_center = 200.0
            d = 200.0
            tip_speed = 50.0  # pixels per second
        else:
            x_center = 0.0
            y_center = 0.0
            d = 0.0075
            tip_speed = 0.003  # meters per second

        p, r = get_square_waypoints(x_center, y_center, d, z=0.03)
        result = mover.servo_arm(
            args.arm, 
            p, 
            args.frame, 
            tool_rotation=r, 
            tip_speed=tip_speed, 
            stop_on_touch=args.stop_on_touch
        )
    else:
        p, r = get_square_waypoints(x_center=0.0, y_center=-0.003, d=0.005, z=0.025)
        result = mover.servo_both_arms(
            p_left=p, 
            r_left=r, 
            frame_left="smoother_uterus/left/base",
            p_right=p,
            r_right=r,
            frame_right="smoother_uterus/right/base", 
            stop_on_touch=args.stop_on_touch
        )

    plot_result(result)

    mover.go_home()
    mover.destroy_node()

    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Visual servo trajectory tester")

    parser.add_argument("--mode", choices=["single", "both"], default="single")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--frame", default="smoother_uterus/camera")
    parser.add_argument("--stop-on-touch", action="store_true")

    args = parser.parse_args()

    run(args)
    