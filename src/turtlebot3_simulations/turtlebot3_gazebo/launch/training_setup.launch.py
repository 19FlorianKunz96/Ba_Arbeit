from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PythonExpression
import os
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

    ld=LaunchDescription()

    #-----------------------------dir----------------------------------------------------------------------
    # gazebo_launch_dir = PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'),'launch'])

    gazebo_launch_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'),'launch')

    #------------------------------------------------Command Line Parameters------------------------------------------------
    #Declare
    stage_arg = DeclareLaunchArgument('stagex', default_value='1', description='number of the trainend stage/world')
    max_episodes_arg = DeclareLaunchArgument('max_episodes',default_value='100', description='maximum of episodes to train')
    stage_boost_arg = DeclareLaunchArgument('stage_boost', default_value='False', description='use weights of pretrained model')
    load_from_folder_arg = DeclareLaunchArgument('load_from_folder', default_value = 'actual', description = 'plug in foldername for training on older data')
    load_from_stage_arg=DeclareLaunchArgument('load_from_stage', default_value='1',description='plug in start of training stage')
    load_from_episode_arg = DeclareLaunchArgument('load_from_episode', default_value = '50', description = 'plug in start of training episode')
    ld.add_action(stage_arg)
    ld.add_action(max_episodes_arg)
    ld.add_action(stage_boost_arg)
    ld.add_action(load_from_folder_arg)
    ld.add_action(load_from_stage_arg)
    ld.add_action(load_from_episode_arg)

    #To be set in command line --> stage:=n
    stage = LaunchConfiguration('stagex')
    max_episodes = LaunchConfiguration('max_episodes')
    stage_boost= LaunchConfiguration('stage_boost')
    load_from_folder=LaunchConfiguration('load_from_folder')
    load_from_stage=LaunchConfiguration('load_from_stage')
    load_from_episode= LaunchConfiguration('load_from_episode')

    #------------------------------------Launch Files------------------------------------------
    # launch1 = IncludeLaunchDescription(PathJoinSubstitution([gazebo_launch_dir,'turtlebot3_dqn_stage1.launch.py',]))
    launch1=IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_launch_dir,'turtlebot3_dqn_stage1.launch.py')),
                                                                    condition=IfCondition(PythonExpression(["'", stage, "' == '1'"])))
    
    launch2=IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_launch_dir,'turtlebot3_dqn_stage2.launch.py')),
                                                                    condition=IfCondition(PythonExpression(["'", stage, "' == '2'"])))
    
    launch3=IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_launch_dir,'turtlebot3_dqn_stage3.launch.py')),
                                                                    condition=IfCondition(PythonExpression(["'", stage, "' == '3'"])))
    
    launch4=IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_launch_dir,'turtlebot3_dqn_stage4.launch.py')),
                                                                    condition=IfCondition(PythonExpression(["'", stage, "' == '4'"])))
    launch_list  =[launch1,launch2,launch3,launch4]
    for launch in launch_list:
        ld.add_action(launch)

    #----------------------------------------Nodes---------------------------------------------------------------------------
    node1= Node(package = 'turtlebot3_dqn',executable = 'dqn_gazebo',name = 'LocationGoalInit',output = 'screen',parameters=[{'stagex':stage}],)
    node2= Node(package = 'turtlebot3_dqn',executable = 'dqn_environment',name ='Environmet',output = 'screen')
    node3= Node(package = 'turtlebot3_dqn',executable='dqn_agent',name='Agent',output = 'screen',parameters=[{'stagex':stage,
                                                                                                              'max_episodes':max_episodes,
                                                                                                              'stage_boost':stage_boost,
                                                                                                              'load_from_folder':load_from_folder,
                                                                                                              'load_from_stage' : load_from_stage,
                                                                                                              'load_from_episode' : load_from_episode}],)
    node4= Node(package = 'turtlebot3_dqn',executable='action_graph',name='ActionGraph',output = 'screen',)
    node5= Node(package = 'turtlebot3_dqn',executable='result_graph',name='ResultGraph',output = 'screen',)
    node6= Node(package = 'turtlebot3_dqn',executable='loss_graph',name='LossGraph',output = 'screen',)

    node_list=[node1,node2,node3,node4,node5,node6]
    for node in node_list:
        ld.add_action(node)

    return ld