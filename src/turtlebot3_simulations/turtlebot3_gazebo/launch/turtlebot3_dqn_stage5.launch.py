import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription,TimerAction,DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    name = 'tb3_house_demo_crowd'
    pedsim_dir = get_package_share_directory('pedsim_simulator')
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    default_pedsim_config_path = os.path.join(pedsim_dir, 'config', 'params.yaml')

    default_pedsim_scene_path = os.path.join(pedsim_dir, 'scenarios', name + '.xml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-3.0')
    y_pose = LaunchConfiguration('y_pose', default='-2.0')

    pedsim_scene_file = LaunchConfiguration('pedsim_scene_file')
    namespace = LaunchConfiguration('namespace')
    pedsim_config_file = LaunchConfiguration('pedsim_config_file')

    declare_pedsim_scene_file_cmd = DeclareLaunchArgument(
        'pedsim_scene_file', 
        default_value=default_pedsim_scene_path,
        description='')
    
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')
    declare_pedsim_config_file_cmd = DeclareLaunchArgument(
        'pedsim_config_file', 
        default_value=default_pedsim_config_path,
        description='')
    
    world = os.path.join(
        get_package_share_directory('pedsim_gazebo_plugin'),
        'worlds',
        name+'.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    agent_spawner_cmd = Node(
        package='pedsim_gazebo_plugin',
        executable='spawn_pedsim_agents',
        name='spawn_pedsim_agents',
        output='screen')
    
    pedsim_launch_cmd = TimerAction(
        period=5.0, # wait for simulator until launching pedsim
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                pedsim_dir, 'launch', 'simulator_launch.py')),
        launch_arguments={
          'scene_file': pedsim_scene_file,
          'config_file': pedsim_config_file,
          'namespace': namespace,
          'use_rviz': 'True'}.items())
        ])

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(declare_pedsim_scene_file_cmd)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_pedsim_config_file_cmd)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(pedsim_launch_cmd)
    ld.add_action(agent_spawner_cmd)

    return ld