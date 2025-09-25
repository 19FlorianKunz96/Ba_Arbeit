import gymnasium as gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class RosGazeboEnv(gym.Env):
    """
    Gym-Environment für lokale Pfadplanung in ROS + Gazebo.
    Aktionen: diskret (z.B. vorwärts, zurück, links, rechts, 45° links/rechts, stehen).
    Beobachtungen: Lidar (downsampled), relative Zielpose, aktuelle Geschwindigkeit.
    """

    def __init__(self):
        super(RosGazeboEnv, self).__init__()

        # --- Action Space ---
        # 0 
        # 1: zurück
        # 2: links drehen
        # 3: rechts drehen
        # 4: 45° links vorwärts
        # 5: 45° rechts vorwärts
        # 6: stehen bleiben
        self.action_space = spaces.Discrete(7)

        # --- Observation Space ---
        # Beispiel: 36 Lidar-Samples + (dx, dy, dtheta) + (v,w)
        lidar_size = 36
        obs_size = lidar_size + 5
        high = np.full((obs_size,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # --- ROS Interfaces ---
        rospy.init_node('ros_gazebo_gym_env', anonymous=True)

        self.cmd_pub = rospy.Publisher('/cmd_vel_rl', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self._scan_cb)
        rospy.Subscriber('/odom', Odometry, self._odom_cb)

        self.scan_data = None
        self.odom = None

        # Lokales Ziel (relativ zur Roboterbasis)
        self.local_goal = np.array([1.0, 0.0, 0.0])  # dx, dy, dtheta

        self.max_steps = 200
        self.current_step = 0

    # --- ROS Callbacks ---
    def _scan_cb(self, msg: LaserScan):
        self.scan_data = msg

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    # --- Gym API ---
    def reset(self):
        """Reset-Simulation + Env-State."""
        self.current_step = 0

        # TODO: hier Gazebo resetten oder Pose setzen
        # z.B. gazebo_reset_world(), set_robot_pose_gazebo()

        # Lokales Ziel setzen
        self.local_goal = np.array([1.0, 0.0, 0.0])

        rospy.sleep(1.0)
        return self._get_obs()

    def step(self, action):
        self.current_step += 1

        # Aktion in Twist übersetzen
        twist = Twist()
        if action == 0:  # vorwärts
            twist.linear.x = 0.2
        elif action == 1:  # zurück
            twist.linear.x = -0.2
        elif action == 2:  # links
            twist.angular.z = 0.5
        elif action == 3:  # rechts
            twist.angular.z = -0.5
        elif action == 4:  # 45° links vorwärts
            twist.linear.x = 0.15
            twist.angular.z = 0.3
        elif action == 5:  # 45° rechts vorwärts
            twist.linear.x = 0.15
            twist.angular.z = -0.3
        elif action == 6:  # stehen
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)
        rospy.sleep(0.2)

        obs = self._get_obs()
        reward, done = self._compute_reward_done(obs)

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    # --- Hilfsfunktionen ---
    def _get_obs(self):
        # LiDAR downsamplen
        if self.scan_data is None:
            lidar = np.ones(36) * 5.0
        else:
            ranges = np.array(self.scan_data.ranges)
            ranges[np.isnan(ranges)] = self.scan_data.range_max
            ranges = np.clip(ranges, 0, self.scan_data.range_max)
            step = len(ranges) // 36
            lidar = np.array([np.min(ranges[i:i+step]) for i in range(0, len(ranges), step)])
            lidar = lidar[:36]

        # Lokales Ziel
        dx, dy, dtheta = self.local_goal

        # Geschwindigkeit (wenn verfügbar)
        v, w = 0.0, 0.0
        if self.odom:
            v = self.odom.twist.twist.linear.x
            w = self.odom.twist.twist.angular.z

        obs = np.concatenate([lidar, [dx, dy, dtheta, v, w]])
        return obs.astype(np.float32)

    def _compute_reward_done(self, obs):
        reward = 0.0
        done = False

        # --- Reward shaping ---
        # Annäherung ans Ziel
        dist_to_goal = np.linalg.norm(obs[-5:-2])  # dx, dy
        reward += -dist_to_goal

        # Kollision (LiDAR unter 0.2m)
        if np.min(obs[:36]) < 0.2:
            reward -= 10.0
            done = True

        # Ziel erreicht
        if dist_to_goal < 0.2:
            reward += 20.0
            done = True

        # Timeout
        if self.current_step >= self.max_steps:
            done = True

        return reward, done

    # --- Hilfsmethoden für Orchestrator ---
    def set_local_goal(self, dx, dy, dtheta=0.0):
        """Setze relatives Ziel (z.B. vom globalen Planner)."""
        self.local_goal = np.array([dx, dy, dtheta])
