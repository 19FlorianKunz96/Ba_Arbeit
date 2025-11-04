'''Environment with State Space28, new Reward,..'''

import math
import os
import time

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal
from copy import deepcopy
#from sklearn.preprocessing import MinMaxScaler


ROS_DISTRO = os.environ.get('ROS_DISTRO')


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 800
        self.last_state = []
        self.current_state = []

        self.done = False
        self.fail = False
        self.succeed = False
        self.collission = False
        self.timeout = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.scan_angles = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.is_front_min_actual_front = False

        self.call=0
        self.last_call=0

        self.max_world_distance = 5.657
        self.last_distance = 1.0

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        self.get_logger().info('Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[25]
        self.prev_goal_distance = self.init_goal_distance
        self.last_distance = self.init_goal_distance
        response.state = state

        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

#---------------------------------get the sensor data from gazebo----------------------------------------------------------------
    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.scan_angles = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        self.front_distance = scan.ranges[0]

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)
            self.scan_angles.append(angle)
            

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        self.min_obstacle_distance_index = int(numpy.argmin(self.scan_ranges))
        self.min_obstacle_angle = self.scan_angles[self.min_obstacle_distance_index]
        self.min_obstacle_distance = self.scan_ranges[self.min_obstacle_distance_index]
        
        self.front_min_obstacle_distance = min(self.front_ranges) if self.front_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        #self.goal_distance_prev=
        self.goal_angle = goal_angle

#-------------------------------------------------calculate the state from sensor data------------------------------------------------------
    def calculate_state(self):
        state = []
        for n, var in enumerate(self.scan_ranges):
            if n % 2 == 0:
                state.append(float(var))
        state.append(float(self.goal_angle))
        state.append(float(self.goal_distance))
        state.append(float(self.min_obstacle_distance))
        state.append(float(self.min_obstacle_distance_index))
        self.local_step += 1

        if self.goal_distance < 0.20:
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            self.collission = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            self.timeout = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        return state
    
#------------------------------------------------reward functions----------------------------------------------------------------
 
    
    def discretize_state(self, state, res=0.02):  # 2 cm Raster
        return tuple(round(x / res) for x in state)


    def calculate_reward(self,action):
        
        # if self.discretize_state(self.last_state) == self.discretize_state(self.current_state):
        #     state_reward = -1
        # else:
        #     state_reward = 0.1

        #yaw_reward = 1 - (2 * abs(self.goal_angle) / math.pi)
        # progress = self.last_distance - self.goal_distance
        #progress_reward = max(min(raw_prog / 0.006, 1.0), -1.0) 
        # self.last_distance =  self.goal_distance

        '''yaw_reward wie im repo'''
        yaw_rewards = []
        for i in range(5):
            angle = -math.pi/4 + self.goal_angle + (math.pi/8 * i) + math.pi/2
            tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * (angle % (2 * math.pi)) / math.pi)[0])
            yaw_rewards.append(tr)
        yaw_reward = yaw_rewards[action]

        progress_rate = 2** (self.goal_distance/self.init_goal_distance)

        if self.min_obstacle_distance <= 0.5:
            obstacle_reward = -5.0
        else:
            obstacle_reward = 1.0

        reward = ((round(yaw_reward * 5, 2)) * progress_rate) + obstacle_reward
        #reward = (yaw_reward * progress_rate) + obstacle_reward

        if self.succeed:
            reward = 1000.0

        elif self.fail:
            reward = -500.0
        print('reward: %f' % (reward))
        return reward
    
#--------------------------------callbacks-----------------------------------------------------------------
    def rl_agent_interface_callback(self, request, response):
        
        # self.call= time.time()
        # self.get_logger().info(f"RL service: {1/ (self.call-self.last_call)}Hz, local_step={self.local_step}")
        # self.last_call=self.call

        action = request.action
        if ROS_DISTRO == 'humble':
            msg = Twist()
            if action == 5:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            else:
                msg.linear.x = 0.2
                msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            if action == 5:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            else:
                msg.linear.x = 0.2
                msg.angular.z = self.angular_vel[action]

        self.cmd_vel_pub.publish(msg)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        self.current_state = self.calculate_state()
        response.state = self.current_state
        response.reward = self.calculate_reward(action=action)
        self.last_state=deepcopy(self.current_state)
        response.done = self.done
        response.success = self.succeed
        response.collission = self.collission
        response.timeout = self.timeout

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False
            self.collission = False
            self.timeout = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()