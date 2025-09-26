#to interact with ros 
import rclpy
from rclpy.node import Node

#how to transport the data ( nothing lost, just the last 10,....)
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile

#message types for the data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


#ToD0:
#Logging only if position is critical for crash to implement emergency stop

class OdomSubscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            qos_profile_sensor_data
        )
        self.subscription

        self.last_time = self.get_clock().now()
        self.interval = 1

    def listener_callback(self, msg: Odometry):

        current_time = self.get_clock().now()
        elapsed = (current_time - self.last_time).nanoseconds / 1e9

        if elapsed >= self.interval:
            self.get_logger().info(
                f'Recieved Odom Header: {msg.header},'
                f'pose: {msg.pose} '
                f'twist: {msg.twist}'
            )
            self.last_time = current_time
        else:
            pass

def main(args=None):
    rclpy.init(args = args)
    node = OdomSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
