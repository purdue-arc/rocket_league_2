import rclpy
from rktl_interfaces.msg import CarAction

def main():
    rclpy.init()
    node = rclpy.create_node("single_msg_node")
    def broadcast():
            node.get_logger().info("sending message")
            send = CarAction()
            send.id = 0
            send.throttle = 1.0
            send.steer = 1.0
            pub.publish(send)
    pub = node.create_publisher(CarAction, "aiTopic", 10)
    broadcast()
    node.destroy_node()
    rclpy.shutdown()