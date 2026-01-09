#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Info
#----------------------------------------------------------------------------------------------------------------------------------------------#
# Anahnd von Odom Daten lokalisieren funktioniert.
# Karten können im Docker Container mit SDF2MAP ohne SLAM generiert werden


#TODO:
# Metriken Plot
# #Metriken :
# Effizienz: globaler Pfad/wirklich gefahrener Pfad
# Success Rate und collission Rate
# Zeit: globaler Pfad / Zeit
# Plot live
# evtl auch wege aufzeichnen?.

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


class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator')
        
        self.classic_mode = False
        self.metrics = dict()

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Feedback + Visualisierung
#----------------------------------------------------------------------------------------------------------------------------------------------#


    #Feedback
        self.feedback=None
        self.last_feedback_time = 0.0

        self.n_runs = 5
        self.current_run= 0
        self.success_counter = 0
        self.collission_counter = 0
        self.current_start_time = None

        self.collision_detector = False

        self.current_path_length = dict()     # Pfadlänge für aktuellen Run
        self.plan_locked = False   # nach neuem Goal erst neuen Plan abwarten
        self.goal_sent_stamp = None

        self.traveled_dist = 0.0
        self._last_odom_xy = None

    #Visualisierung
        plt.ion()
        self.fig,self.ax = plt.subplots()

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Trajektorienplanung
#----------------------------------------------------------------------------------------------------------------------------------------------#

        self.stage = 4

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
        self.save_log_info = f'/Stage{self.stage}_Planner_.csv' #Auch action space , welcher agent, klassisch, ... angeben
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Services + Topics + Actions
#----------------------------------------------------------------------------------------------------------------------------------------------#

    #Topics

        self.odom_sub = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        self.amcl_counter = 0
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.on_scan, 1)
        self.collision_publisher = self.create_publisher(Bool,'/collision_stop',10)
        self.plan_topic = '/plan' if self.classic_mode else '/rl/target_path'
        self.sub_global_path = self.create_subscription(Path, self.plan_topic, self.on_global_path, 10)

    #Services
        self.respawn_client = self.create_client(SetEntityState, '/set_entity_state')
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.reset_global_costmap_client = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')
        self.reset_local_costmap_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.lifecycle_manager = self.create_client(ManageLifecycleNodes,'/lifecycle_manager_navigation/manage_nodes')
        self.load_map_client = self.create_client(LoadMap,'/map_server/load_map')

        self.path_to_map = '/home/verwalter/maps/turtlebot3_dqn_stage4.yaml'



    #Actions
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          INIT - Functions
#----------------------------------------------------------------------------------------------------------------------------------------------#
        self.nav_goal_handle = None
        self.is_resetting = False
        
        self.init_timer = self.create_timer(1, self.init_amcl)
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Class - Functions
#----------------------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                Logging
#----------------------------------------------------------------------------------------------------------------------------------------------#

########################################################## Distance ############################################################################

    def path_length(self, path_msg: Path) -> float:
        poses = path_msg.poses
        if len(poses) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(poses)):
            p1 = poses[i-1].pose.position
            p2 = poses[i].pose.position
            length += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return float(length)

    def on_global_path(self, msg: Path):
        # Optional: nur akzeptieren, wenn wir gerade auf einen Plan warten
        if self.plan_locked:
            #self.get_logger().info("ignored: plan_locked=True")
            return
        
        if not getattr(self,'wait_for_first_plan',False):
            self.get_logger().info("ignored: wait_for_first plan")
            return
        
        if len(msg.poses)<2:
            return

        self.current_path_length[self.current_run]=self.path_length(msg)
        self.plan_locked = True
        self.wait_for_first_plan = False

        self.get_logger().info(f"Planned path length (Run {self.current_run}): {self.current_path_length[self.current_run]:.2f} m")

    def odom_callback(self,msg:Odometry):
        self.init_act_pose = (msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

    # traveled distance integrieren (nur wenn ein Run aktiv ist)
        if self.current_start_time is not None:
            if self._last_odom_xy is not None:
                lx, ly = self._last_odom_xy
                self.traveled_dist += math.hypot(x - lx, y - ly)
            self._last_odom_xy = (x, y)
        else:
            self._last_odom_xy = (x, y)

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

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                               
#----------------------------------------------------------------------------------------------------------------------------------------------#
    def on_scan(self, msg:LaserScan):
        for distance in msg.ranges:
            if distance < 0.15:
                msg = Bool()
                msg.data = True
                self.collision_publisher.publish(msg)
                self.collision_detector = True
                self.collision_elimination()
                break
        

    def init_amcl(self):
        self.publish_initial_pose()
        self.amcl_counter +=1
        if self.amcl_counter > 2:
            self.amcl_counter = 0
            self.init_timer.destroy()
            self.send_next_goal()

    def collision_elimination(self):
        self.get_logger().warn(f'Collision detected... Starting Respawn')
        self.collission_counter+=1

        #0. aktives Goal canceln
        self.cancel_active_goal()
        #1. Roboter in Gazebo respawnen
        target_pose = self.respawn_robot()
        #self.reset_simulation() #alternativ ganze simulation restetten. Dann respawnt der Roboter aber bei x=0.0, y=0.0
        time.sleep(3) #Vorher 3 Sekunden TEST

        # self.reload_map()
        # time.sleep(0.2)

        if self.classic_mode:
            self.clear_global_costmap()
            time.sleep(2)

            self.clear_local_costmap()
            time.sleep(2)

        # self.reset_nav2()
        # time.sleep(3)

        # self.startup_nav2()
        # time.sleep(3)

        #2. AMCL Pose neu setzen mit neu gespawnter Roboterpose
        self.relocalize_amcl(target_pose)
        #self.relocalize_amcl(self.current_goal.pose)
        time.sleep(2) #vorher 2 Sekunden TEST



        #3. Goal neu setzen
        self.send_next_goal()
        self.get_logger().warn(f'Ready !!!')

        self.collision_detector = False
        msg = Bool()
        msg.data = self.collision_detector
        self.collision_publisher.publish(msg)

    
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


    def respawn_robot(self):
        #neues Programm: hier soll in Zukunft immer das angefahrene Ziel bei einer Kollision zum neuen Startpunkt werden
        self.get_logger().info("Set Entity...")
        state = EntityState()
        state.name='waffle_pi'

        target_pose = self.current_goal.pose
        state.pose = target_pose
        req = SetEntityState.Request()
        req.state = state
        future = self.respawn_client.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=1.0)

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

    def send_next_goal(self):

        if self.current_run >= self.n_runs:
            df = pd.DataFrame.from_dict(self.metrics,orient='index')
            df.index.name = 'RUN_ID'
            df.to_csv(self.folder_path+self.save_log_info, sep=';', index=True, index_label='RUN_ID')
            rclpy.shutdown()
            return
        
        goal = self.create_goal()
        self.current_goal = goal
        self.current_run +=1

        msg = NavigateToPose.Goal()
        msg.pose = goal
        
        self.get_logger().info(f'neues Goal gesetzt')
        self.current_start_time = time.monotonic()

        self.traveled_dist = 0.0
        self._last_odom_xy = None
        #self.current_path_length = None
        self.plan_locked = False #vorher True
        self.wait_for_first_plan = False


        send_goal_future = self._action_client.send_goal_async(msg,feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self,future):###
        goal_handle = future.result()
        if not goal_handle.accepted:
            # self.collission_counter += 1
            # self.send_next_goal()
            return
        
        self.nav_goal_handle = goal_handle
        self.plan_locked = False
        self.wait_for_first_plan = True

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def cancel_active_goal(self):
        if self.nav_goal_handle is None:
            return

        self.get_logger().warn("Cancel actual Goal...")
        cancel_future = self.nav_goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
        self.nav_goal_handle = None
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                               Result Callback
#                                                     # hier wird auch Effizienz, etc. berechnet #
#----------------------------------------------------------------------------------------------------------------------------------------------#

    def result_callback(self, future):
        self.metrics[self.current_run] = {
                                    'success/collision': None,
                                    'planned_path': None,
                                    'traveled_path': None,
                                    'time_taken': None,
                                            }

        self.get_logger().info('result_callback aufgerufen')
        end_time = time.monotonic()
        duration = end_time - self.current_start_time if self.current_start_time else 0.0

        status = future.result().status

        #planned = self.current_path_length[self.current_run]
        planned = self.current_path_length.get(self.current_run, None)
        traveled = self.traveled_dist
        if planned is None:
            self.get_logger().warn(f'No Planned Path for this Run -> Collision/Canceled')
            self.get_logger().warn(f'Current stats: success={self.success_counter}, 'f'failure={self.collission_counter}')
            self.metrics[self.current_run]['success/collision'] = 'collision'
        else:
        # Beispiel-Effizienz: planned / traveled (1.0 ideal, <1 Umwege, >1 selten)
            eff_path = planned / traveled if traveled > 1e-6 else float('inf')
            eff_time = traveled / duration
            self.get_logger().warn(f"Run {self.current_run}: planned={planned:.2f} m, traveled={traveled:.2f} m, time={duration:.2f} s, eff(planned/traveled)={eff_path:.3f}, eff(time)={eff_time}, time={duration:.2f} s")
            self.metrics[self.current_run]['planned_path'] = planned
            self.metrics[self.current_run]['traveled_path'] = traveled
            self.metrics[self.current_run]['time_taken'] = duration
            self.metrics[self.current_run]['success/collision'] = 'success'
            self.metrics[self.current_run]['path_efficiency'] = eff_path
            self.metrics[self.current_run]['time_effiency'] = eff_time


        if status == GoalStatus.STATUS_SUCCEEDED:
            self.success_counter += 1
            self.get_logger().info(f'Goal #{self.current_run} SUCCEEDED in {duration:.2f} s 'f'(status={status}).')
            if hasattr(self,'current_goal'):
                self.last_safe_goal = PoseWithCovarianceStamped()
                self.last_safe_goal.header.frame_id = self.frame_id
                self.last_safe_goal.pose.pose = self.current_goal.pose

            self.get_logger().info(f'Current stats: success={self.success_counter}, 'f'failure={self.collission_counter}')
            self.send_next_goal()


    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback




def main():
    rclpy.init()
    node = Evaluator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()