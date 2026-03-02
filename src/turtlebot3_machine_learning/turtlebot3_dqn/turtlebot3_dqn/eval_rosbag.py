#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Program
#----------------------------------------------------------------------------------------------------------------------------------------------#

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Message Types
#----------------------------------------------------------------------------------------------------------------------------------------------#
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from turtlebot3_msgs.srv import Dqn
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.msg import GoalState
from std_srvs.srv import Empty
from nav2_msgs.srv import LoadMap
from nav2_msgs.srv import ClearEntireCostmap
from nav2_msgs.srv import ManageLifecycleNodes
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from std_msgs.msg import Int8
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Standart Libraries
#----------------------------------------------------------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import time
import math
import random
import csv
import rclpy.time
import pandas as pd
from datetime import datetime
import subprocess, signal, os
import json
from std_msgs.msg import String

from pathlib import Path as SysPath

class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator')
        
        self.classic_mode = False
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Feedback + Visualisierung
#----------------------------------------------------------------------------------------------------------------------------------------------#

        # Segment / Event logging
        self.event_pub = self.create_publisher(String, '/perf/segment_event', 10)

        self.feedback=None
        self.last_feedback_time = 0.0

        self.n_runs = 15
        self.current_run= 1
        self.success_counter = 0
        self.collission_counter = 0
        self.collision_detector = False

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Zielplanung
#----------------------------------------------------------------------------------------------------------------------------------------------#

        self.stage = 1

        if self.stage == 5:
            self.goals = [(1.0,1.0),(3.9,-3.15),(-3.0,-2.0),(-6.0,-2.6),(-2.4,3.2),(6.0,3.0),(6.5,-3.5),(0.8,3.0),(-0.5,-1.4)]

        elif self.stage == 4:
            self.goals = [(1.0, 0.0), (2.0, -1.5), (0.0, -2.0), (2.0, 1.5), (0.5, 2.0), (-1.5, 2.1),(-2.0, 0.5), (-2.0, -0.5), (-1.5, -2.0), (-0.5, -1.0), (2.0, -0.5), (-1.0, -1.0)]

        else:

            self.goals = [(-1.1, -0.2), (-0.6, 0.5), (-1.0, 0.8), (-2.0, -0.8), (0.1, 1.7), (1.7, 0.8), (0.1, -0.4), (-1.2, -1.6), (-1.8, -0.5), (-0.8, 1.8)]

        

        self.max_goals = 100
        self.frame_id = 'map'

        self.last_safe_pose = PoseWithCovarianceStamped()
        self.last_safe_pose.pose.pose.position.x,self.last_safe_pose.pose.pose.position.y = self.goals[0]
        self.last_safe_goal = PoseWithCovarianceStamped()
        self.last_safe_goal.pose.pose.position.x,self.last_safe_goal.pose.pose.position.y = self.goals[0]
        self.start_pose = PoseWithCovarianceStamped()
        self.start_pose.pose.pose.position.x,self.start_pose.pose.pose.position.y = self.goals[0]    
        self.init_act_pose = (0.0,0.0,1.0,0.0,0.0,0.0)

        self.folder_path = '/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log'
        self.save_log_info = f'/Stage{self.stage}_AS5_RewardRobotis_AlleKomponenten_Vortrainiert' #Auch action space , welcher agent, klassisch, ... angeben
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Services + Topics + Actions
#----------------------------------------------------------------------------------------------------------------------------------------------#

    #Topics
        self.amcl_counter = 0
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.on_scan, 1)
        self.collision_publisher = self.create_publisher(Bool,'/collision_stop',10)
        self.plan_topic = '/plan' if self.classic_mode else '/rl/target_path'

    #Services
        self.respawn_servicename = '/gazebo_spawner/set_entity_state' if self.stage==5 else '/set_entity_state'
        self.respawn_client = self.create_client(SetEntityState, self.respawn_servicename)
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.reset_global_costmap_client = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')
        self.reset_local_costmap_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.lifecycle_manager = self.create_client(ManageLifecycleNodes,'/lifecycle_manager_navigation/manage_nodes')
        self.load_map_client = self.create_client(LoadMap,'/map_server/load_map')

        self.path_to_map = '/home/verwalter/maps/turtlebot3_dqn_stage4.yaml'



    #Actions
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          INIT - Timers
#----------------------------------------------------------------------------------------------------------------------------------------------#

        self.start_rosbag_all()
        
        self.init_timer = self.create_timer(1, self.init_amcl)
        self.nav_goal_handle=None
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Helper - Functions
#----------------------------------------------------------------------------------------------------------------------------------------------#

    def init_amcl(self):
        self.publish_initial_pose()
        self.amcl_counter +=1
        if self.amcl_counter > 2:
            self.amcl_counter = 0
            self.init_timer.destroy()
            self.send_next_goal()
    
    def create_goal(self) -> PoseStamped:
        x,y = self.goals[self.current_run % len(self.goals)]

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        return pose

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                Logging
#----------------------------------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                              collision detection/elimination                 
#----------------------------------------------------------------------------------------------------------------------------------------------#
    def on_scan(self, msg:LaserScan):

        min_range = min(msg.ranges) if msg.ranges else float('inf')
        if min_range < 0.15 and not self.collision_detector:

            self.collision_detector = True
            self.emit("COLLISION", self.current_run, min_range=float(min_range))

            msg = Bool()
            msg.data = True
            self.collision_publisher.publish(msg)
            self.collision_elimination()

    def collision_elimination(self):

        self.get_logger().warn(f'Collision detected... Starting Respawn')

        #0. aktives Goal canceln
        self.cancel_active_goal()

        #1. Roboter in Gazebo respawnen
        target_pose = self.respawn_robot()
        time.sleep(3)

        if self.classic_mode:
            self.clear_global_costmap()
            self.clear_local_costmap()

        #2. AMCL Pose neu setzen mit neu gespawnter Roboterpose
        self.relocalize_amcl(target_pose)
        time.sleep(2)

        self.get_logger().warn(f'Ready !!!')
        self.collision_detector = False
        msg = Bool()
        msg.data = self.collision_detector
        self.collision_publisher.publish(msg)

    



    def respawn_robot(self):

        self.get_logger().info("Set Entity...")
        state = EntityState()
        state.name='waffle_pi'

        target_pose = self.current_goal.pose
        state.pose = target_pose
        req = SetEntityState.Request()
        req.state = state
        future = self.respawn_client.call_async(req)
        timeout=3 if self.stage==5 else 1.0
        rclpy.spin_until_future_complete(self,future,timeout_sec=timeout)

        if self.stage == 5:
            pass

        else:
            try:
                result = future.result()
            except Exception as e:
                self.get_logger().error(f"Respawn-Service fehlgeschlagen: {e}")
            else:
                self.get_logger().info("Respawn in Gazebo OK.")

        return target_pose
    
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                            Goals   
#----------------------------------------------------------------------------------------------------------------------------------------------#

    def send_next_goal(self):
       
        goal = self.create_goal()
        self.current_goal = goal
        run_id = self.current_run

        self.emit("RUN_START",run_id,goal_xy=[goal.pose.position.x, goal.pose.position.y])


        msg = NavigateToPose.Goal()
        msg.pose = goal
        self.get_logger().info(f'neues Goal gesetzt')

        #Goal wird gesendet und goal_response_callback wird mit goal_response_future aufgerufen, sobald Nav2 annimmt oder ablehnt
        #feedback_callback wird zyklisch aufgerufen, immer wenn ein feedback von Nav2 gesendet wird
        send_goal_future = self._action_client.send_goal_async(msg,feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(lambda fut: self.goal_response_callback(fut,run_id))

        return True

    def goal_response_callback(self,future,run_id):
        #wenn das goal nicht angenommen wurde: Funktion verlassen
        goal_handle = future.result()
        if not goal_handle.accepted:
            #self.current_run += 1
            return
        
        self.nav_goal_handle = goal_handle

        #fragt nach, wann das Ziel fertig ist.
        result_future = goal_handle.get_result_async()

        #result_callback wird aufgerufen bei succeeded, aborted, canceled oder Fehler im ActionServer
        result_future.add_done_callback(lambda fut : self.result_callback(fut,run_id))

    def cancel_active_goal(self):
        if self.nav_goal_handle is None:
            return

        self.get_logger().warn("Cancel actual Goal...")

        #aktuelles goal wird im action server gecanceled
        cancel_future = self.nav_goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)

        #init, dass gerade kein goal gehandled wird
        self.nav_goal_handle = None
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                               Result Callback
#                                           Das Result-Callback wird aufgerufen, wenn ein Ziel erreicht ist. 
#----------------------------------------------------------------------------------------------------------------------------------------------#


    def result_callback(self, future, run_id):

        #ROSBAG
        status = future.result().status
        self.emit("RUN_END", run_id, status=int(status))

        if status != GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(f"{run_id} not Successfull")
            self.collission_counter += 1
 
        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(f"{run_id} Successfull")
            self.success_counter += 1

        if self.current_run >= self.n_runs:
            self.stop_rosbag()
            rclpy.shutdown()

        #WICHTIG!!! Neues Ziel erst wenn Ergebnis da
        else:
            self.current_run += 1
            self.send_next_goal()
        return


    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                   Nav2 Services            
#----------------------------------------------------------------------------------------------------------------------------------------------#

    def reset_nav2(self):
        req = ManageLifecycleNodes.Request()
        req.command = 3
        future = self.lifecycle_manager.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=10.0)
        self.get_logger().warn("NAV2 Reset done...")

    def startup_nav2(self):
        req = ManageLifecycleNodes.Request()
        req.command = 0
        future = self.lifecycle_manager.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=10.0)
        self.get_logger().warn("NAV2 Startup done...")
    
    def clear_local_costmap(self):
        req = ClearEntireCostmap.Request()
        future = self.reset_local_costmap_client.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=3.0)
        self.get_logger().warn("Clear Local Costmap done...")

    def clear_global_costmap(self):
        req = ClearEntireCostmap.Request()
        future = self.reset_global_costmap_client.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=3.0)
        self.get_logger().warn("Clear Global Costmap done...")

    def reload_map(self):
        req = LoadMap.Request()
        req.map_url=self.path_to_map
        future = self.load_map_client.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=10.0)
        self.get_logger().warn("Reload Map done...")

    def reset_simulation(self):
        self.get_logger().info("Reset Simulation...")
        req = Empty.Request()
        for i in range(2):
            self.publish_initial_pose()
            time.sleep(0.2)
        future = self.reset_client.call_async(req)
        self.get_logger().info("Reset done...")   
        rclpy.spin_until_future_complete(self,future,timeout_sec=10.0)


    def publish_initial_pose(self):
        if self.init_act_pose is None:
            self.get_logger().warn("init_pose is None. Waiting for Gazebo pose")
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rclpy.time.Time().to_msg()

        msg.pose.pose.position.x, msg.pose.pose.position.y,msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z = self.init_act_pose

        # kleine Kovarianz nötig. nur so groß dass sie als gültig angesehen wird
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685
        self.initial_pose_pub.publish(msg)
        time.sleep(1)
        self.get_logger().info("Initial pose published!")

#################################################ROSBAG#############################################################################################################################
    def relocalize_amcl(self,pose):
        self.get_logger().info("Set AMCL...")
        time.sleep(1)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = pose
        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.1

        for i in range(2):
            self.initial_pose_pub.publish(msg)
            time.sleep(0.2)

    def start_rosbag_all(self):
        if getattr(self, "bag_process", None) is not None:
            return

        base = SysPath("/home/verwalter/rosbags")
        base.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.save_log_info.lstrip("/")
        bag_name = str(base / f"{safe_name}_{ts}")

        topics = [
            "/scan", "/tf", "/tf_static", "/odom", "/amcl_pose", "/cmd_vel",self.plan_topic,
            "/navigate_to_pose/_action/goal",
            "/navigate_to_pose/_action/cancel",
            "/navigate_to_pose/_action/status",
            "/navigate_to_pose/_action/feedback",
            "/navigate_to_pose/_action/result",
            "/perf/segment_event",   # (empfohlen) deine Run-Marker
            "/collision_stop"        # optional
        ]

        cmd = ["ros2", "bag", "record", "-o", bag_name] + topics
        self.get_logger().info(f"Starting rosbag for ALL runs: {bag_name}")

        self.bag_process = subprocess.Popen(cmd, start_new_session=True)

    def stop_rosbag(self):
        if getattr(self, "bag_process", None) is None:
            return

        self.get_logger().info("Stopping rosbag...")
        os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
        try:
            self.bag_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.bag_process.kill()
            self.bag_process.wait()
        self.bag_process = None

    def emit(self, event: str, run_id: int, **fields):

        payload = {
            "event": event,
            "run_id": int(run_id),
            "ros_time_ns": int(self.get_clock().now().nanoseconds),
            **fields
        }
        msg = String()
        msg.data = json.dumps(payload)#, ensure_ascii=False)
        self.event_pub.publish(msg)




def main():
    rclpy.init()
    node = Evaluator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()