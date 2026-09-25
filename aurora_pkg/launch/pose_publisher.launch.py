from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('ndi_aurora_interface'),
        'config',
        'pose_publisher_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='ndi_aurora_interface',
            executable='pose_publisher',
            name='aurora_pose_publisher',
            output='screen',
            parameters=[config]
        )
    ])
