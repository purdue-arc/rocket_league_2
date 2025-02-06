
import rclpy.node
from rktl_simulator.simulatorPoint import PointGame
from rktl_simulator.simulator import FIELD_WIDTH, GOAL_DEPTH, FIELD_HEIGHT
from rktl_interfaces.msg import Field, CarAction, Pose
import rclpy
from rclpy.node import Node
class tester(Node):
        def __init__(self):
            super().__init__("tester")
            self.recievedMessage = False
            self.subscription_ = self.create_subscription(Field, "simTopic", self.printFieldState, 10)
            self.publisher_ = self.create_publisher(CarAction, "aiTopic", 10)
        
        def printFieldState(self, msg):
            self.recievedMessage = True
            self.get_logger.info("recieved:")
            self.get_logger.info(f"Ball Pose: x:{msg.ball_pose.x}, y:{msg.ball_pose.y}, id:{msg.ball_pose.id}")
            self.get_logger.info(f"Car Pose: x:{msg.team1_poses[0].x}, y:{msg.team1_poses[0].y}, angle: {msg.team1_poses[0].angle_degrees}, id:{msg.team1_poses[0].id}")
            self.broadcast()
            
        def broadcast(self):
            self.get_logger().info("sending message")
            send = CarAction()
            send.id = 0
            send.throttle = 1.0
            send.steer = 1.0
            self.publisher_.publish(send)
        

def main(args = None):
    import time

    rclpy.init()
    testerNode = tester()
    rclpy.spin(testerNode)
    testerNode.destroy_node()
    rclpy.shutdown()