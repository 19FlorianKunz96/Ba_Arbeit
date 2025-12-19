#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Info
#----------------------------------------------------------------------------------------------------------------------------------------------#
# Anahnd von Odom Daten lokalisieren funktioniert.
# Karten können im Docker Container mit SDF2MAP ohne SLAM generiert werden

#TODO: 

##!!!!!!!!!!!!!!!!!!!!!  Goals müssen so generiert werden, dass es für jede map passt !!!!!!!!!!!!!
##Vll Liste für jede Map? irgendwie mit YAML File ? 

#Grafik erstellen wie die Nodes in Nav2-Gazebo-Agent-Evaluation zusammenhängen und interagieren
#Das selbe vll auch noch für den Trainingsagenten

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
from turtlebot3_msgs.msg import GoalState

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Standart Libraries
#----------------------------------------------------------------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import time
import math
import random


class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator')
#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Feedback + Visualisierung
#----------------------------------------------------------------------------------------------------------------------------------------------#
    #Feedback
        self.feedback=None
        self.last_feedback_time = 0.0

        self.n_runs = 15
        self.current_run= 0
        self.success_counter = 0
        self.collission_counter = 0
        self.current_start_time = None

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
        self.init_act_pose = None


#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Services + Topics + Actions
#----------------------------------------------------------------------------------------------------------------------------------------------#
    
        self.callback_group = ReentrantCallbackGroup()
    #Topics

        self.odom_sub = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        self.amcl_counter = 0
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,'/initialpose',10)
        self.timer = self.create_timer(1, self.publish_initial_pose)

    #Services
        self.respawn_client = self.create_client(SetEntityState, '/set_entity_state',callback_group = self.callback_group)
        self.collission_detection_service = self.create_service(Dqn, '/collission_detection',self.collission_callback,callback_group=self.callback_group)


    #Actions
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          INIT - Functions
#----------------------------------------------------------------------------------------------------------------------------------------------#
        self.nav_goal_handle = None
        self.is_resetting = False
        

#----------------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Class - Functions
#----------------------------------------------------------------------------------------------------------------------------------------------#


    def collission_callback(self,req,resp):
        
        self.collission_counter += 1 
        self.get_logger().info(f'Status: success={self.success_counter}, 'f'failure={self.collission_counter}')

        ok = self.respawn_robot()
        if ok:
            self.get_logger().info("Robot successfully respawned")
        else:
            self.get_logger().error("Respawn failed")
        resp.done = True
        return resp
    
    def publish_initial_pose(self):
        if self.init_act_pose is None:
            self.get_logger().warn("init_pose is None. Waiting for Gazebo pose")
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x, msg.pose.pose.position.y,msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z = self.init_act_pose

        # kleine, gültige Kovarianz
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685

        if self.amcl_counter == 0:
            self.initial_pose_pub.publish(msg)
            self.get_logger().info("Initial pose published!")

        self.amcl_counter += 1

        if self.amcl_counter > 2:
            self.timer.cancel()
            self.send_next_goal()
            self.amcl_counter = 0


    def odom_callback(self,msg:Odometry):

        # aktuelle Pose wird immer empfangen -> kann später auch für etwas anderes genutzt werden. 
        # Momentan nur für Initialpose beim start der Evaluation verwendet für AMCL
         self.init_act_pose = (msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.orientation.w,msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z)

    def relocalize_amcl(self,pose):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = pose

        msg.pose.covariance[0] = 0.05  # x
        msg.pose.covariance[7] = 0.05  # y
        msg.pose.covariance[35] = 0.1  # yaw

        for i in range(3):
            self.initial_pose_pub.publish(msg)
            time.sleep(1)
        self.get_logger().info(f"AMCL Initialpose gesendet.")

        return True


    def respawn_robot(self):

        #self.cancel_navigation_blocking(timeout_sec=1.0)
        #globalen pfad neu berechnen und an agent schicken ?
        if self.last_safe_pose is None and getattr(self, "start_pose", None) is not None:
            self.last_safe_pose = self.start_pose

        if self.last_safe_pose is None and self.last_safe_goal is None:
            self.get_logger().warn("Kein gültiges Respwanziel vorhanden.")
            return
        
        state = EntityState()
        state.name='waffle_pi'

        if self.last_safe_goal is not None:
            target_pose = self.last_safe_goal.pose.pose
        else:
            target_pose = self.last_safe_pose   #.pose.pose
        
        state.pose = target_pose

        req = SetEntityState.Request()
        req.state = state

        future = self.respawn_client.call_async(req)
        rclpy.spin_until_future_complete(self,future,timeout_sec=2.0)
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Respawn-Service fehlgeschlagen: {e}")

        else:
            self.get_logger().info("Respawn in Gazebo OK.")
        
        time.sleep(1)
        ok = self.relocalize_amcl(target_pose)
        while ok != True:
            self.get_logger().info('Locating')
            time.sleep(0.1)

        ok = False
        time.sleep(1)
        ok = self.send_next_goal()

        while ok != True:
            self.get_logger().info('Sending Goal')
            time.sleep(0.1)

        return True



    def create_goal(self) -> PoseStamped:
        x,y = self.goals[self.current_run % len(self.goals)]

        yaw = random.uniform(-math.pi, math.pi)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def send_next_goal(self):

        if self.current_run >= self.n_runs:
            ##################################
            ##################################
            ##### Hier Logging + Save Data ###
            ##################################
            ##################################

            rclpy.shutdown()
            return
        
        goal = self.create_goal()
        self.current_goal = goal
        self.current_run +=1

        msg = NavigateToPose.Goal()
        msg.pose = goal
        
        self.get_logger().info(f'neues Goal gesetzt')
        self.current_start_time = time.monotonic()

        send_goal_future = self._action_client.send_goal_async(msg,feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self,future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.collission_counter += 1
            self.send_next_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self.get_logger().info('result_callback aufgerufen')
        end_time = time.monotonic()
        duration = end_time - self.current_start_time if self.current_start_time else 0.0

        goal_result = future.result()
        status = goal_result.status
        result_msg = goal_result.result

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.success_counter += 1
            self.get_logger().info(f'Goal #{self.current_run} SUCCEEDED in {duration:.2f} s 'f'(status={status}).')
            if hasattr(self,'current_goal'):
                self.last_safe_goal = PoseWithCovarianceStamped()
                self.last_safe_goal.header.frame_id = self.frame_id
                self.last_safe_goal.pose.pose = self.current_goal.pose

            self.get_logger().info(f'Current stats: success={self.success_counter}, 'f'failure={self.collission_counter}')
            self.send_next_goal()

        else:
            self.collission_counter += 1
            self.get_logger().warn(f'Goal #{self.current_run} FAILED (nav2 result={result_msg}, 'f'action status={status}) after {duration:.2f} s.')
            self.get_logger().info(f'Current stats: success={self.success_counter}, 'f'failure={self.collission_counter}')
            self.respawn_robot()


    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback
        #self.get_logger().info(f'distance_remaining={feedback.distance_remaining:.2f}')




def main():
    rclpy.init()
    node = Evaluator()
    executor = MultiThreadedExecutor(num_threads = 4)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()