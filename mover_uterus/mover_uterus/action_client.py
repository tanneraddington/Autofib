import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from aliss_ros_msg.action import VisualServo
import subprocess
from aliss_ros_msg.msg import VisualServoWaypoint, VisualServoGoalTraj


def log_goal_traj(traj):
    details = f"num_waypoints: {len(traj.waypoints)}, frame_id: {traj.frame_id}\n"
    for wp in traj.waypoints:
        details += f"  time: {wp.time_seconds:.3f}, pos: ({wp.x:.4f}, {wp.y:.4f}, {wp.z:.4f}), rot: {wp.tool_rotation:.3f}\n"
    return details


def get_goal_traj(position, tool_rotation, frame, tip_speed, tip_speed_z_limit=0.003):
    # Broadcast tool rotation to position shape if currently a scalar
    if isinstance(tool_rotation, (float, int)):
        tool_rotation = [tool_rotation] * len(position)

    traj = VisualServoGoalTraj()

    for p, r in zip(position, tool_rotation):
        waypoint = VisualServoWaypoint()
        waypoint.x = p[0]
        waypoint.y = p[1]
        waypoint.z = p[2]
        waypoint.tool_rotation = r
        # Don't fill out waypoint.time_seconds here. Server will do that
        traj.waypoints.append(waypoint)
    
    traj.frame_id = frame
    traj.tip_speed = tip_speed
    traj.tip_speed_z_limit = tip_speed_z_limit

    return traj


class SmootherMover(Node):

    def __init__(self):
        super().__init__(f'visual_servo_action_client')
        self.client = ActionClient(self, VisualServo, '/visual_servoing/visual_servo')
        
    def send_goal_and_wait(self, goal):
        self.get_logger().info(f'Waiting for action server...')
        self.client.wait_for_server()

        self.get_logger().info(f'Sending goal for visual servoing:')
        self.get_logger().info(f'LEFT arm:  {log_goal_traj(goal.left_goal_traj)}\n')
        self.get_logger().info(f'RIGHT arm: {log_goal_traj(goal.right_goal_traj)}\n')

        send_future = self.client.send_goal_async(goal)  # (feedback_callback=...) optional
        
        self.get_logger().info('Goal sent, waiting for response...')
        # rclpy.spin_until_future_complete(self, send_future)
        self._servo_executor.spin_until_future_complete(send_future)
        
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return None
        
        self.get_logger().info('Goal accepted.')
        result_future = goal_handle.get_result_async()

        self.get_logger().info('Waiting for visual servoing result...')
        # rclpy.spin_until_future_complete(self, result_future)
        self._servo_executor.spin_until_future_complete(result_future)

        result = result_future.result().result   # has .status and .result

        self.get_logger().info(
            f"Result: success={result.success} "
            f"stopped_timeout={getattr(result,'stopped_timeout',False)} "
            f"stopped_touched={getattr(result,'stopped_touched',False)}"
        )

        return result
    
    def servo_both_arms(self,
                   p_left=None,
                   r_left=0.0,
                   frame_left='',
                   p_right=None,
                   r_right=0.0,
                   frame_right='',
                   tip_speed=0.001,
                   stop_on_touch=False,
                   cautery_mode=0,
                   cautery_arm='right'):

        goal = VisualServo.Goal()
        goal.stop_on_touch = stop_on_touch
        goal.cautery_mode = int(cautery_mode)
        goal.cautery_arm = cautery_arm

        if p_left is None:
            goal.left_goal_traj = VisualServoGoalTraj()
        else:
            goal.left_goal_traj = get_goal_traj(p_left, r_left, frame_left, tip_speed)
        
        if p_right is None:
            goal.right_goal_traj = VisualServoGoalTraj()
        else:
            goal.right_goal_traj = get_goal_traj(p_right, r_right, frame_right, tip_speed)

        return self.send_goal_and_wait(goal)

    def servo_arm(self, arm, points, frame, tool_rotation=0.0, tip_speed=0.003,
                  stop_on_touch=False, cautery_mode=0, cautery_arm='right'):
        goal = VisualServo.Goal()
        goal.stop_on_touch = stop_on_touch
        goal.cautery_mode = int(cautery_mode)
        goal.cautery_arm = cautery_arm

        goal_traj = get_goal_traj(points, tool_rotation, frame, tip_speed=tip_speed)
        if arm == 'left':
            goal.left_goal_traj = goal_traj
            goal.right_goal_traj = VisualServoGoalTraj()
        elif arm == 'right':
            goal.left_goal_traj = VisualServoGoalTraj()
            goal.right_goal_traj = goal_traj
        else:
            raise ValueError("arm must be 'left' or 'right'")
        
        return self.send_goal_and_wait(goal)

    def go_home(self):
        # Uterine robot: joint command 0 is the motor_node home, both arms in one message
        # [R_ir, R_or, R_it, R_ot, L_ir, L_or, L_it, L_ot]. (NOT /set_home, which re-zeros home at the
        # current pose.) `ros2 topic pub -1` waits for motor_node to match before publishing.
        subprocess.run(["ros2", "topic", "pub", "-1", "/robot/state/current_state",
                        "std_msgs/msg/Float32MultiArray", "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"])


