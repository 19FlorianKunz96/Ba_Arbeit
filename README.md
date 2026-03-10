# Reinforcement Learning for TurtleBot3

This project implements and evaluates several **Deep Q-Network (DQN) extensions** for autonomous navigation using **TurtleBot3** in **ROS2** and **Gazebo**.  
The trained agent can also be integrated as a **local planner within Nav2**.

---

# Requirements

- ROS2 (Humble recommended)
- Gazebo
- TurtleBot3 packages
- Python machine learning dependencies (depending on your implementation)

Make sure the TurtleBot3 environment variable is set before running the simulation:

```bash
export TURTLEBOT3_MODEL=waffle_pi
```

If another robot model is used, adjust the value accordingly.

If this variable is not set correctly, Gazebo may fail to load the robot and the node may hang.

---

# Training the Agent

Training is started using a ROS2 launch file.

## Start Training

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py
```

---

## Training Parameters

The following parameters can be set when starting the training:

| Parameter | Description | Default |
|----------|-------------|--------|
| stagex | Defines the training stage (which Gazebo world is used) | 1 |
| max_episodes | Number of training episodes | 100 |
| action_space | Size of the action space | 5 |

Example:

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py stagex:=4 max_episodes:=4000
```

---

## Training Environments

Different environments can be selected in the launch file.

Available environments:

**custom_environment.py**  
Baseline environment used for the initial agent.

**new_environment.py**  
Environment described in the research paper.

If `new_environment.py` is used, the parameter **state_size must be adjusted in the agent**.

---

## Training with Pretrained Weights (Stage Boost)

Training can continue using weights from a previous training stage.

| Parameter | Description | Default |
|----------|-------------|--------|
| stage_boost | Enables loading pretrained weights | False |
| load_from_folder | Folder containing the pretrained model | actual |
| load_from_stage | Stage from which the model was trained | 1 |
| load_from_episode | Episode used to load weights | 100 |

Example:

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py \
stagex:=4 \
max_episodes:=4000 \
stage_boost:=True \
load_from_folder:=<path_to_pretrained_model> \
load_from_stage:=<stage_number> \
load_from_episode:=<episode_number>
```

Make sure the pretrained model uses **the same architecture**.

---

## Training Outputs

During training the following data is saved **every 50 episodes**:

- model weights
- epsilon values
- training configuration
- training graphs

Results are stored in the following directory:

```
trainings_done/{uuid}_{date}_{stage}
```

inside the machine learning package.

---

# Testing the Agent

The trained agent can be evaluated using a dedicated test launch file.

## Configure Model Parameters

Before testing, ensure the following parameters match the trained model:

- state_size
- action_size

These parameters must be set manually in the **test agent**.

---

## Run the Test

```bash
ros2 launch turtlebot3_gazebo agent_test.launch.py \
stagex:=<testing_stage> \
max_episodes:=<number_of_test_episodes> \
load_from_folder:=<model_folder> \
load_from_stage:=<training_stage> \
load_from_episode:=<trained_episode> \
action_space:=<action_space>
```

Default number of testing episodes: **100**.

---

# Using the Agent in Nav2

The reinforcement learning agent can also be used as a **local planner in Navigation2**.

---

## Local Planner Configuration

Currently, the RL agent is integrated as the **default local planner**.

If you want to use the classic **DWB local planner**, start Nav2 with the following parameter file:

```
/home/verwalter/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble/waffle_pi_classic.yaml
```

---

## Start the Simulation

```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage<number>.launch.py
```

---

## Start Navigation2

```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
use_sim_time:=True \
map:=<path_to_map> \
params_file:=<path_to_parameter_file_if_classic_mode_is_used>
```

---

## Start the RL Agent

```bash
ros2 run turtlebot3_dqn nav2_agent
```

If a **real TurtleBot3** is used, set the following parameter in `nav2_agent.py`:

```
real_mode = true
```

---

# Automated Evaluation with Nav2

Automated testing can be performed using the evaluation node.

## Start Simulation and Navigation

Start the same components as described in **Using the Agent in Nav2**.

---

## Start Evaluation Node

```bash
ros2 run turtlebot3_dqn evaluation
```

This node performs automated evaluation of the agent's performance.

---

# Project Structure (Example)

```
turtlebot3_dqn/
│
├── environments/
│   ├── custom_environment.py
│   └── new_environment.py
│
├── agents/
│   ├── training_agent.py
│   └── test_agent.py
│
├── launch/
│   ├── training_complete.launch.py
│   └── agent_test.launch.py
│
├── nav2_agent.py
├── evaluation.py
│
└── trainings_done/
```

---

# Notes

- Ensure that **action_space and state_size match the trained model**.
- Pretrained models must use **the same network architecture**.
- Gazebo may hang if the `TURTLEBOT3_MODEL` environment variable is not set.

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

