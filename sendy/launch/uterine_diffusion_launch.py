from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sendy',
            executable='uterine_diffusion_node',
            name='uterine_diffusion_node',
            output='screen',
        ),
        Node(
            package='sendy',
            executable='uterine_move_node',
            name='uterine_move_node',
            output='screen',
        ),
        Node(
            package='sendy',
            executable='uterine_move_node_r',
            name='uterine_move_node_r',
            output='screen',
        ),
        # task_publisher_node is interactive (reads stdin) -- usually launched
        # separately in its own terminal rather than via this launch file,
        # but included here for convenience if you want everything in one go.
        Node(
            package='sendy',
            executable='uterine_task_publisher',
            name='uterine_task_publisher',
            output='screen',
            prefix='xterm -e',
        ),
    ])
