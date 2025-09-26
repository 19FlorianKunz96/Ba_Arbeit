import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


#ToD0:
#Logging only if position is critical for crash to implement emergency stop

class LidarSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile_sensor_data
        )
        self.subscription

        self.last_time = self.get_clock().now()
        self.interval = 10

    def listener_callback(self, msg: LaserScan):

        current_time = self.get_clock().now()
        elapsed = (current_time - self.last_time).nanoseconds / 1e9

        if elapsed >= self.interval:
            self.get_logger().info(
                f'Header: {msg.header}'
                f'Recieved Scan with: {len(msg.ranges)} ranges,'
                f'minimum range: {msg.range_min:.2f} m'
            )
            self.last_time = current_time
        else:
            pass

def main(args=None):
    rclpy.init(args = args)
    node = LidarSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
