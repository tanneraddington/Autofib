import math
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def is_true(value):
    return value.lower() in ('1', 'true', 'yes', 'y', 't')

def generate_launch_description():

    run_dir = DeclareLaunchArgument(
        'run_dir',
        default_value='',
        description='The directory to save the run logs'
    )

    load_calib = DeclareLaunchArgument(
        'load',
        default_value='0',
        description='load:=1 Load smoother calibration file (default: 0)'
    )

    # Define a flag to determine whether to launch visual_servoing node
    run_visual_servo_server = DeclareLaunchArgument(
        'run_visual_servo_server',
        default_value='true',
        description='Whether to launch visual_servoing nodes'
    )

    run_rviz = DeclareLaunchArgument(
        'run_rviz',
        default_value='true',
        description='Whether to launch RViz display',
    )

    left_tool_length_arg = DeclareLaunchArgument(
        'left_tool_length',
        default_value='0.005',
        description='left tool (tip) length in meters, default is spatula (0.004)'
    )

    right_tool_length_arg = DeclareLaunchArgument(
        'right_tool_length',
        default_value='0.008',
        description='right tool (tip) length in meters, default is electrosurgery (0.008)'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_rect',
        description='RECTIFIED endoscope image (the uterine robot publishes raw frames on /image)'
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='CameraInfo matching image_topic'
    )

    return LaunchDescription([
        load_calib,
        run_visual_servo_server,
        run_rviz,
        run_dir,
        left_tool_length_arg,
        right_tool_length_arg,
        image_topic_arg,
        camera_info_topic_arg,
        OpaqueFunction(function=launch_setup),
    ])

def launch_setup(context, *args, **kwargs):
    
    nodes = []

    load_calib = is_true(LaunchConfiguration('load').perform(context))
    run_visual_servo = is_true(LaunchConfiguration('run_visual_servo_server').perform(context))
    run_rviz = is_true(LaunchConfiguration('run_rviz').perform(context))
    run_dir = LaunchConfiguration('run_dir').perform(context)

    left_tool_length = float(LaunchConfiguration('left_tool_length').perform(context))
    right_tool_length = float(LaunchConfiguration('right_tool_length').perform(context))
    image_topic = LaunchConfiguration('image_topic').perform(context)
    camera_info_topic = LaunchConfiguration('camera_info_topic').perform(context)

    # Create smoother_uterus node with the smoother config.yaml file
    nodes.append(Node(
        package='smoother_uterus',
        executable='smoother_uterus',
        name='smoother_uterus',
        output='screen',
        parameters=[
            {
                'load': load_calib,
                'run_dir': run_dir,
                'left.tip_offset.z': left_tool_length,
                'right.tip_offset.z': right_tool_length,
                'image_topic': image_topic,
                'camera_info_topic': camera_info_topic,
            },
        ],
    ))

    if run_visual_servo:
        nodes.append(Node(
            package='visual_servoing',
            executable='action_server',
            name='action_server',
            output='screen',
            prefix='taskset -c 0',  # TODO figure out how to not do it this way
            parameters=[{'camera_info_topic': camera_info_topic}],
        ))

    # motor_node's /fwkin_<l,r> (and the diffusion trajectories built from them) live in each arm's
    # own FK frame: z along the arm, outer_rot = 0 bends toward +x. The smoother's arm base frame
    # bends toward -y at outer_rot = 0, so the two differ by a -90 deg yaw about z (same origin).
    # This lets TF / the mover servo on diffusion trajectories: frame_id 'hy/<side>/fwkin'.
    for side in ('left', 'right'):
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'fwkin_{side}_tf',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', str(-math.pi / 2), '--pitch', '0', '--roll', '0',
                '--frame-id', f'smoother_uterus/{side}/base',
                '--child-frame-id', f'hy/{side}/fwkin',
            ],
        ))

    if run_rviz:
        pkg_share = get_package_share_directory('smoother_uterus')
        rviz_config_file = os.path.join(pkg_share, 'config', 'smoother.rviz')
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
        ))


    return nodes
