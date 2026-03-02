# RL-Learning for Turtlebot3

## Training(all extensions of the DQN) with Launchfile
1. set environtment variable "TURTLEBOT3_MODEL:=wallfe_pi" in bash file (if other robots are used, adjust the value) to load the robot in gazebo, otherwise the node will hang and the world will not load correctly in future
2. The environment can be exchanged by selecting another environment node in the launchfile. custom_environment.py is the environment from the baseline agent.
   new_environment.py is the environment from paper. The parameter state_size has to be adjusted in the agent
3. starting launchfile in x-terminal-emulator with 'ros2 launch turtlebot3_gazebo training_complete.launch.py
   - here you can set some parameters:
     - with stagex:= you can set the training stage, in wich world the robot will learn. Default value is 1 and it is requiered to plug in an integer value
     - with max_episodes:= you can set the number of episodes, the robot will learn. Default value is 100 and its requiered to plug in an integer value
     - with action_space:= you can set the expanded actionspace(default:5)-> if this is set to 6, the parameter action_space has to be adjusted in the environment
     - example prompt:
         ```console
         $ ros2 launch turtlebot3_gazebo training_complete.launch.py stagex:=4 max_episodes:=4000
     - with stage_boost:= the robot will not learn with zero weights, but will load weights from an older stage. Epsilon will be set to 0.2. Default value is False and its required to plug in an Boolean value
     - with load_from_folder:= (just rational if stage_boost is set to True) you can plug in the basename of an path from an older training(folder 'trainings_done'), to load the weights. Default value is 'actual', so it will load theyoungest weights of the folder 'saved_model'
     - with load_from_stage:= you can set the stage(normally the stage from the foldername).Default value is 1 and its required to plug in an integer value.
     - with load_from_episode:= you can adjust the episode for loading weights(just rational if load_from_folder is not 'actual').Default value is 100 and its required to plug in an integer value.
     - if stageboost: Make sure, the the pretrained model is from the same architecture
         ```console
         $ ros2 launch turtlebot3_gazebo training_complete.launch.py stagex:=4 max_episodes:=4000 stage_boost:=True load_from_folder<path to pretrained model folder> load_from_stage:=<the stage the predtrained model was trained> load_from_episode:=<epsisode the training should start>
4. after training the weights(every 50 episodes), the epsilons, a config file from the training and the graphs are stored in the machine learning package in a folder 'trainings_done/{uuid}_{date}_{stage}'

## Test the agent
1. As in the training the environment can be exchanged
2. the state_size and the action_size has to be manually set in the testagent
   ```console
   $ ros2 launch turtlebot3_gazebo agent_test.launch.py stagex:=<testing stage> max_episodes:=<number of testig episodes(default:100)> load_from_folder<path to pretrained model folder> load_from_stage:=<the stage the predtrained model was trained> load_from_episode:=<epsisode of the trained model for testing> action_space:=<action space of the trained model(default:5)>


## Use the agent in Nav2
1. The agent is included as standard local planner at the moment. If the classic local planner DWB should be used, the parameterfile /home/verwalter/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble/waffle_pi_classic.yaml
 has to be set in the prompt when starting Nav2
2. Start the Simulation or the Bringup for real Turtlebot. If the real turtlebot is used, the parameter real_mode has to be set to true in nav2_agent.py
   ```console
   $ ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage<number of the world to be loaded>.launch.py
3. Start Nav2 with the correct map
   ```console
   $ ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=<path to map> params_file:=<path to parameter file if classic mode is used>
3. Start the Agent
   ```console
   $ ros2 run turtlebot3_dqn nav2_agent

## Automated Tests with Nav2

1. 
2. Start the same files as in "Use the agent in Nav2"
3. Start the Evaluation Node
   ```console
   $ ros2 run turtlebot3_dqn evaluation

