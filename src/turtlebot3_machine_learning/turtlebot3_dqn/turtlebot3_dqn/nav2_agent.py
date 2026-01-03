
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Info
#----------------------------------------------------------------------------------------------------------------------------------------------#
#               Dieser Node stellt ausschließlich nur noch den Service zur Auswahl des Steuerbefehls zur Verfügung und
#               unterteilt einen globalen Pfad in kleine Teilziele.

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Program
#----------------------------------------------------------------------------------------------------------------------------------------------#
import os
from pathlib import Path as FSPath
import math
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Load ROS Components
#----------------------------------------------------------------------------------------------------------------------------------------------#
import rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from turtlebot3_msgs.srv import RLLocalPath
from geometry_msgs.msg import PoseStamped
from turtlebot3_msgs.msg import GoalState
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from turtlebot3_msgs.srv import Dqn
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                       init Quality of Service for AMCL
#----------------------------------------------------------------------------------------------------------------------------------------------#

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Load the Agent Classes
#----------------------------------------------------------------------------------------------------------------------------------------------#

from turtlebot3_dqn.utils import DQNMetric
from turtlebot3_dqn.utils import Dueling_DQN
from turtlebot3_dqn.utils import PERBuffer
from turtlebot3_dqn.utils import N_Step
from turtlebot3_dqn.utils import NoisyDense
from turtlebot3_dqn.utils import DuelingQRDQN
from turtlebot3_dqn.utils import quantile_huber_loss
from turtlebot3_dqn.utils import euler_from_quaternion

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                              Load Tensorflow
#----------------------------------------------------------------------------------------------------------------------------------------------#
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class DQNadvisor(Node):
    def __init__(self):
        super().__init__('dqn_advisor')
        self.folder_path= FSPath('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation/PER_N2_D3QN/c8afb22e-058d-444b-8e3e-3a218ac2eed4_2025-11-03_stage4_rainbow')
        self.episode = 'stage00004_episode04000.h5'
        self.model_path = os.path.join(self.folder_path,self.episode)
        self.declare_parameter('service_name','/agents/dqn/get_action')
        self.declare_parameter('state_size',28)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        self.action_size = 5

        self.test_mode = True
        self.rainbowmode=False
        self.distributional_mode=False
        self.full_noisy_dense = True
        self.num_quantiles=51

        self.init_pose = None

        self.state_size = self.get_parameter('state_size').get_parameter_value().integer_value
        self.obs = np.zeros((1, self.state_size), dtype = np.float32)

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.sub_path = self.create_subscription(Path, '/rl/target_path', self.on_path, 1)
        self.sub_collsion = self.create_subscription(Bool, '/collision_stop', self.on_collision_stop, 1)

        self.pub_goal = self.create_publisher(GoalState, 'goal', 10)

        srv_name = self.get_parameter('service_name').value
        self.srv = self.create_service(RLLocalPath, srv_name, self.on_service_handle)

        self.collission_client = self.create_client(Dqn, '/collission_detection')

        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        self.scan_ranges = []
        self.scan_angles = []
        self.front_ranges = []
        self.front_angles = []
        self.local_goal = None
        self.robot_pose_x = self.robot_pose_y = 0.0
        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.min_obstacle_distance = 10.0
        self.min_obstacle_distance_index = 0
        self.global_path = []
        self.last_index=0
        self.done=True
        self.reached = False

        self.collision_in_progress = False
        self.collission_detection = False
        self.collision_publishing = False

        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #                                                         Get the Model
        #----------------------------------------------------------------------------------------------------------------------------------------------#

        

        if str(self.folder_path).endswith('rainbow') and self.distributional_mode:
            self.rainbowmode = True
            try:
                self.model.load_model(self.model_path, compile=False, custom_objects = {'Dueling_QR:DQN':DuelingQRDQN, 'NoisyDense':NoisyDense})
            except Exception:
                self.model= DuelingQRDQN(n_actions=self.action_size,fc1=256,fc2=256,num_quantiles=self.num_quantiles,full_noisy=self.full_noisy_dense)
                _ = self.model(tensorflow.zeros((1,self.state_size),dtype=tensorflow.float32))
                self.model.load_weights(self.model_path)

        elif str(self.folder_path).endswith('rainbow'):
            self.rainbowmode=True
            try:
                self.model.load_model(self.model_path, compile = False, custom_objects={'Dueling_DQN':Dueling_DQN, 'NoisyDense':NoisyDense})
            except Exception:
                self.model = Dueling_DQN(self.action_size,fc1=256,fc2=256, full_noisy = self.full_noisy_dense)
                _ = self.model(tensorflow.zeros((1, self.state_size),dtype=tensorflow.float32))
                self.model.load_weights(self.model_path)


        else:
            self.model = self.build_model()
            loaded_model = load_model(self.model_path, compile=False, custom_objects={'mse': MeanSquaredError()})
            self.model.set_weights(loaded_model.get_weights())

        #----------------------------------------------------------------------------------------------------------------------------------------------#
        #                                                         Functions
        #----------------------------------------------------------------------------------------------------------------------------------------------#
    #Hört ob eine Kollision bepublisht wird
    def on_collision_stop(self,msg:Bool):
        self.collision_publishing = msg.data
        if self.collision_publishing:
            self.get_logger().warn(f'Stop... Collision...')
        else:
            self.get_logger().warn(f'Moving... Collision fixed...')


    def on_scan(self, msg:LaserScan):
        self.scan_ranges = []
        self.scan_angles = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        self.front_distance = msg.ranges[0]

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = msg.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif np.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)
            self.scan_angles.append(angle)
            

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        self.min_obstacle_distance_index = int(np.argmin(self.scan_ranges))
        self.min_obstacle_angle = self.scan_angles[self.min_obstacle_distance_index]
        self.min_obstacle_distance = self.scan_ranges[self.min_obstacle_distance_index]
        
        self.front_min_obstacle_distance = min(self.front_ranges) if self.front_ranges else 10.0

    def on_odom(self, msg: Odometry):
        self.init_act_pose = (msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = euler_from_quaternion(msg.pose.pose.orientation)

        if not self.global_path:
            return
        
        if self.local_goal is None or self.done:
            self.sub_target()
            if self.local_goal is None:
                return
#-----------------------------------------------------------------------------------------------------#
        self.goal_pose_x, self.goal_pose_y = self.local_goal
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
        self.goal_angle = goal_angle
#------------------------------------------------------------------------------------------------------#
#Wenn ein Subtarget erreicht wurde
        if goal_distance < 0.3:
            if self.last_index == len(self.global_path) - 1:
                self.done=False
            else:
                self.done= True



    def on_path(self, msg:Path):
        self.global_path = msg.poses
        self.last_index = 0
        self.get_logger().info(f'neuer globaler Pfad')

    def sub_target(self):
        if not self.global_path:
            return
        
        if not self.done:
            return
        
        msg = GoalState()
        chosen = False
             
        for n in range(self.last_index+1, len(self.global_path)):
            path = self.global_path[n].pose.position
            x = round(float(path.x),1)
            y = round(float(path.y),1)
            distance =  math.hypot(x-self.robot_pose_x,y-self.robot_pose_y)
                

            if distance > 0.5 and distance < 3:
                self.local_goal = (x,y)
                self.last_index = n
                self.done = False
                chosen = True
                break
            
        if not chosen:
            last = self.global_path[-1].pose.position
            self.local_goal = (round(float(last.x),1), (round(float(last.y),1)))
            self.last_index = len(self.global_path) - 1
            self.done = False

    def build_state(self):
        state = []
        for n, var in enumerate(self.scan_ranges):
            if n % 2 == 0:
                state.append(float(var))
        state.append(float(self.goal_angle))
        state.append(float(self.goal_distance))
        state.append(float(self.min_obstacle_distance))
        state.append(float(self.min_obstacle_distance_index))
        return state
    

    
    def advise(self):
        state = self.build_state()
        
        if self.rainbowmode and self.distributional_mode:
            s=np.asarray(state,dtype=np.float32).reshape(1,-1)
            q_exp = self.model.q_expectation(s, training = False)
            action = int(tensorflow.argmax(q_exp,axis=1).numpy()[0])

        elif self.rainbowmode:
            s=np.asarray(state,dtype=np.float32).reshape(1,-1)
            q=self.model(s,training=False).numpy()[0]
            action = int(np.argmax(q))
        else:
            state = np.asarray(state)
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action = int(np.argmax(q_values[0]))

        velocity_x= 0.2 if action != 5 else 0.0
        velocity_y=0.0
        angular_z = self.angular_vel[action] if action != 5 else 0.0
        return velocity_x,velocity_y,angular_z


    def on_service_handle(self,req,resp):       
        instruction = self.advise()

        if self.collision_publishing:
            resp.ok = True
            resp.vx = 0.0; resp.vy = 0.0; resp.wz = 0.0
            resp.msg = "collision_stop"
            resp.action_stamp = self.get_clock().now().to_msg()
            return resp

        if instruction is None:
            resp.ok = False
            resp.msg = 'No fresh observation'
            return resp
        
        vx, vy, wz = instruction
        # clamp
        vx = float(max(min(vx, req.max_vx), -req.max_vx))
        wz = float(max(min(wz, req.max_wz), -req.max_wz))
        resp.vx, resp.vy,resp.wz = vx,vy,wz
        resp.action_stamp = self.get_clock().now().to_msg()
        resp.ok = True
        resp.msg = ''
        return resp

        

def main():
    rclpy.init()
    node = None
    try:
        dqn_advisor = DQNadvisor()
        rclpy.spin(dqn_advisor)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
