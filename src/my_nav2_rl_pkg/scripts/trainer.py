import rospy
import gym
import numpy as np
from stable_baselines3 import DQN
from ros_gazebo_gym_env import RosGazeboEnv  # unser Environment

def main():
    rospy.init_node('rl_trainer', anonymous=True)

    # --- Environment ---
    env = RosGazeboEnv()

    # --- RL Agent ---
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        tau=1.0,
        target_update_interval=500,
        train_freq=1,
    )

    # --- Training Loop ---
    total_timesteps = 20000
    print("Starting training for {} timesteps...".format(total_timesteps))
    model.learn(total_timesteps=total_timesteps)

    # --- Save Model ---
    model.save("dqn_local_navigation")
    print("Training finished. Model saved as dqn_local_navigation.zip")

    # --- Test Run ---
    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
