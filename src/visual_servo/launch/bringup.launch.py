import os
import time

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

BAG_ROOT = os.path.expanduser(os.environ.get('RP_BAG_DIR', '~/rp_bags'))
BAG_TOPICS = ['/robot_state', '/des_value', '/aruco_detector/pose', '/visual_servo/debug', '/rosout']


def servo_node():
    return Node(package='visual_servo', executable='servo_node', output='screen')


def after_home(event, context):
    if context.perform_substitution(LaunchConfiguration('servo')).lower() != 'true':
        return []
    if event.returncode == 0:
        return [LogInfo(msg='Home reached: starting the visual servo.'), servo_node()]
    return [LogInfo(msg=f'Homing failed (exit {event.returncode}): visual servo NOT started.')]


def start_servo_without_home(context):
    if context.perform_substitution(LaunchConfiguration('servo')).lower() == 'true':
        return [servo_node()]
    return []


def generate_launch_description():
    rviz = LaunchConfiguration('rviz')
    record = LaunchConfiguration('record')
    home = LaunchConfiguration('home')
    bag_dir = LaunchConfiguration('bag_dir')
    home_node = Node(package='visual_servo', executable='home_node', output='screen', condition=IfCondition(home))
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false', description='Show the robot model in RViz'),
        DeclareLaunchArgument('record', default_value='false', description='Record a rosbag of the servo topics'),
        DeclareLaunchArgument('servo', default_value='false', description='Also start the visual servo controller'),
        DeclareLaunchArgument('home', default_value='true', description='Move to the home pose (0, pi/2) at start'),
        DeclareLaunchArgument('bag_dir', default_value=os.path.join(BAG_ROOT, time.strftime('servo_%Y%m%d_%H%M%S')),
                              description='Output folder of the rosbag'),
        DeclareLaunchArgument('command_rate', default_value='0.0',
                              description='Extra joint-command rate to the robot in Hz (0 = stock 10 Hz; try 20)'),
        Node(package='twolink_control', executable='serial_node', output='screen',
             parameters=[{'command_rate': ParameterValue(LaunchConfiguration('command_rate'), value_type=float)}]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py']))),
        Node(package='marker_detection', executable='detection_realsense', output='screen'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('twolink_gazebo'), 'launch', 'gazebo.launch.py'])),
            launch_arguments={'use_gazebo': 'false'}.items(),
            condition=IfCondition(rviz)),
        Node(package='twolink_gazebo', executable='joint_publisher.py', output='screen', condition=IfCondition(rviz)),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', bag_dir] + BAG_TOPICS, output='screen',
                       condition=IfCondition(record)),
        home_node,
        RegisterEventHandler(OnProcessExit(target_action=home_node, on_exit=after_home)),
        OpaqueFunction(function=start_servo_without_home, condition=UnlessCondition(home)),
    ])
