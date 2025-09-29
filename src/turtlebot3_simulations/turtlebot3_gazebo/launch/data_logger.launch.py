from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import ExecuteProcess

#TODO:
# open nodes in different terminals when launching

def generate_launch_description():
    test_arg = LaunchConfiguration('new_name')
    
    return LaunchDescription([

        # ExecuteProcess(
        #     cmd = ['x-terminal-emulator', '--hold','-x','ros2','run','turtlebot3_dqn','lidar_subscriber'],
        #     output = 'screen'
        # ),
        # ExecuteProcess(
        #     cmd = ['x-terminal-emulator','--hold', '-x','ros2','run','turtlebot3_dqn','odom_subscriber'],
        #     output = 'screen'
        # )

        Node(
            package = 'turtlebot3_dqn',
            executable = 'lidar_subscriber',
            name = 'Lidar_Data',
            output = 'screen',
            prefix=['x-terminal-emulator', ' -e', ' bash -c "ros2 run turtlebot3_dqn lidar_subscriber"; exec bash'],
        ),
        Node(
            package = 'turtlebot3_dqn',
            executable = 'odom_subscriber',
            name = 'Odometrie_Data',
            output = 'screen',
            prefix=['x-terminal-emulator', ' -e', ' bash -c "ros2 run turtlebot3_dqn odom_subscriber"; exec bash']
        )
    ])