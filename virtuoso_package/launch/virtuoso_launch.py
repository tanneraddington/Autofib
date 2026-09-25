from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch Python Nodes
        Node(
            package='virtuoso_package',
            executable='Virtuoso_UI_R.py',
            name='virtuoso_ui_r_py',
            output='screen'
        ),
        Node(
            package='virtuoso_package',
            executable='Virtuoso_UI_L.py',
            name='virtuoso_ui_l_py',
            output='screen'
        ),

        # Launch C++ Nodes
        Node(
            package='virtuoso_package',
            executable='virtuoso_ui_node_r',
            name='virtuoso_ui_node_r',
            output='screen'
        ),
        Node(
            package='virtuoso_package',
            executable='virtuoso_ui_node_l',
            name='virtuoso_ui_node_l',
            output='screen'
        ),
        Node(
            package='virtuoso_package',
            executable='combine_control',
            name='combine_control',
            output='screen'
        )
    ])
