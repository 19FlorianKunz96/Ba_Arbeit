# RL-Learning for TurtleBot3

## Training (DQN Extensions) with Launchfile

### 0. Create Training Output Directory

Before starting the training process, ensure that the required output directory exists inside your workspace. The agent stores trained models and results in a subfolder called `trainings_done`.

Create it using:

```bash
mkdir -p <your_workspace>/src/turtlebot3_machine_learning/turtlebot3_dqn/trainings_done
```

### 1. Set environment variable

Set the TurtleBot3 model in the bash environment:

```bash
export TURTLEBOT3_MODEL=waffle_pi
```

If another robot model is used, adjust the value accordingly.

If this variable is not set correctly, the Gazebo node may hang and the world may not load properly in future runs.

---

### 2. Select training environment

The environment can be exchanged by selecting another environment node in the launch file.

Available environments:

- `custom_environment.py`  
  Environment used by the baseline agent.

- `new_environment.py`  
  Environment used in the paper.

If `new_environment.py` is used, the parameter **state_size** has to be adjusted in the agent.

---

### 3. Start training

Run the launch file in the terminal:

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py
```

Available parameters:

| Parameter | Description | Default |
|----------|-------------|--------|
| `stagex` | Training stage (defines which world is used) | 1 |
| `max_episodes` | Number of training episodes | 100 |
| `action_space` | Size of the action space | 5 |

Example:

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py stagex:=4 max_episodes:=4000
```

---

### 4. Training with pretrained weights (stage boost)

The agent can also be trained using weights from a previous training.

| Parameter | Description | Default |
|----------|-------------|--------|
| `stage_boost` | Enables loading pretrained weights | False |
| `load_from_folder` | Folder containing the pretrained model | actual |
| `load_from_stage` | Stage of the pretrained model | 1 |
| `load_from_episode` | Episode used to load weights | 100 |

Example:

```bash
ros2 launch turtlebot3_gazebo training_complete.launch.py \
stagex:=4 \
max_episodes:=4000 \
stage_boost:=True \
load_from_folder:=<path_to_pretrained_model_folder> \
load_from_stage:=<stage_number> \
load_from_episode:=<episode_number>
```

Make sure the pretrained model uses the same architecture.

---

### 5. Training outputs

During training the following data is saved every **50 episodes**:

- model weights
- epsilon values
- training configuration
- graphs

Results are stored in:

```
trainings_done/{uuid}_{date}_{stage}
```

inside the machine learning package.

---

# Testing the Agent

### 1. Configure environment

As in the training process, the environment can be exchanged.

### 2. Adjust model parameters

The following parameters must be set manually in the test agent:

- `state_size`
- `action_size`

These values must match the trained model.

---

### 3. Run the test

```bash
ros2 launch turtlebot3_gazebo agent_test.launch.py \
stagex:=<testing_stage> \
max_episodes:=<testing_episodes> \
load_from_folder:=<model_folder> \
load_from_stage:=<training_stage> \
load_from_episode:=<trained_episode> \
action_space:=<action_space>
```

Default number of test episodes: **100**

---

# Using the Agent in Nav2

### 1. Local planner configuration

Currently the RL agent is included as the **default local planner**.

If the classic DWB planner should be used, the following parameter file must be set when starting Nav2:

```
/home/verwalter/turtlebot3_ws/src/turtlebot3/turtlebot3_navigation2/param/humble/waffle_pi_classic.yaml
```

---

### 2. Start simulation

Start the simulation world:

```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage<number>.launch.py
```

---

### 3. Start Nav2

```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
use_sim_time:=True \
map:=<path_to_map> \
params_file:=<parameter_file_if_classic_planner_is_used>
```

---

### 4. Start the agent

```bash
ros2 run turtlebot3_dqn nav2_agent
```

If a real TurtleBot is used, set

```
real_mode = true
```

in `nav2_agent.py`.

---

# Automated Tests with Nav2

1. Start the same components as in **Using the Agent in Nav2**.

2. Start the evaluation node:

```bash
ros2 run turtlebot3_dqn evaluation
```

# Agent Parametrization

Some training parameters are not set in the launch file, but directly inside the agent and utility classes.

### Environment-related parameters

These parameters have to match the selected environment:

| Parameter | Description |
|----------|-------------|
| `state_size` | Size of the state vector. Must match the selected environment. |
| `action_space` / `action_size` | Number of available actions. Must match both the environment and the trained model. |

Example:
- `custom_environment.py` uses `state_size = 26`
- `new_environment.py` uses `state_size = 28`

---

### General training parameters

The following parameters are currently set directly in the agent:

| Parameter | Default | Description |
|----------|---------|-------------|
| `discount_factor` | `0.99` | Discount factor for future rewards |
| `learning_rate` | `0.001` | Learning rate for the optimizer |
| `batch_size` | `64` | Batch size used for replay sampling |
| `min_replay_memory_size` | `5000` | Minimum number of samples before training starts |
| `max_buffer_size` | `500000` | Maximum replay buffer size |
| `gradient_clipping` | `True` | Enables gradient clipping during training |

---

### Target network parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `use_soft_target` | `True` | Enables soft target updates |
| `tau` | `0.005` | Update factor for the soft target update |

If soft target updates are disabled, the target network is updated manually after a defined number of steps.

---

### N-step learning parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `nsteps` | `5` | Number of steps used for n-step return calculation |
| `mix_one_step` | `True` | Mixes additional one-step samples into replay |
| `mixing_ratio` | `0.2` | Probability of adding a one-step transition |

---

### Prioritized Experience Replay (PER)

| Parameter | Default | Description |
|----------|---------|-------------|
| `alpha` | `0.6` | Controls how strongly priorities affect sampling |
| `beta_start` | `0.4` | Initial value for importance-sampling correction |
| `beta_frames` | `300000` | Number of frames over which beta increases |
| `eps_per` | `1e-2` | Small value added to priorities for numerical stability |

PER is used to sample more important transitions more often during training.

---

### Network architecture parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `fc1` | `256` | Number of units in the first hidden layer |
| `fc2` | `256` | Number of units in the second hidden layer |
| `fc3` | `None` | Optional third hidden layer |
| `full_noisy_dense` | `True` | Enables NoisyDense layers in the feature network |

The current architecture uses a dueling network structure.

---

### Noisy Nets parameters

Exploration is currently handled with **Noisy Networks**, so no epsilon-greedy strategy is used.

| Parameter | Default | Description |
|----------|---------|-------------|
| `full_noisy_dense` | `True` | Replaces standard dense layers with `NoisyDense` layers |
| `sigma0` | `0.5` | Initial noise scale in `NoisyDense` |

If `training=True`, noise is sampled in the NoisyDense layers.  
If `training=False`, only the deterministic weights are used.

---



### Current training setup

The current agent configuration includes:

- Dueling DQN
- Double DQN target computation
- Prioritized Experience Replay (PER)
- N-step learning
- Noisy Networks
- Soft target updates

