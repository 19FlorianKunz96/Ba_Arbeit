# RL-Learning for Turtlebot3

## Training with Launchfile
1. set environtment variable "TURTLEBOT3_MODEL:=wallfe_pi" in bash file (if other robots are used, adjust the value) to load the robot in gazebo, otherwise the node will hang and the world will not load correctly in future
2. starting launchfile in x-terminal-emulator with 'ros2 launch turtlebot3_gazebo training_setup.launch.py
   - here you can set some parameters:
     - with stagex:= you can set the training stage, in wich world the robot will learn. Default value is 1 and it is requiered to plug in an integer value
     - with max_episodes:= you can set the number of episodes, the robot will learn. Default value is 100 and its requiered to plug in an integer value
     - with stage_boost:= the robot will not learn with zero weights, but will load weights from an older stage. Epsilon will be set to 0.2. Default value is False and its required to plug in an Boolean value
     - with load_from_folder:= (just rational if stage_boost is set to True) you can plug in the basename of an path from an older training(folder 'trainings_done'), to load the weights. Default value is 'actual', so it will load theyoungest weights of the folder 'saved_model'
     - with load_from_stage:= you can set the stage(normally the stage from the foldername).Default value is 1 and its required to plug in an integer value.
     - with load_from_episode:= you can adjust the episode for loading weights(just rational if load_from_folder is not 'actual').Default value is 100 and its required to plug in an integer value.
3. after training the weights(every 50 episodes), the epsilons, a config file from the training and the graphs are stored in the machine learning package in a folder 'trainings_done/{uuid}_{date}_{stage}'

## Test the agent
1. to test the agent you can run the node from the tutorial: 'ros2 run turtlebot3_dqn dqn_test TODO: adjust the file to load from an training from folder'trainings_done''
   - here you can set some parameters:
