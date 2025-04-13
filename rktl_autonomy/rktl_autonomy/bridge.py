import rclpy
from rclpy.node import Node
from rktl_interfaces.msg import Field, CarAction

class Bridge(Node):
    def __init__(self):
        super().__init__("bridge_node")
        self.subscription = self.create_subscription(
            Field,
            'simTopic',
            self.field_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(CarAction, 'aiTopic', 1)
        self.get_logger().info('I am now in the Bridge')

        self.ball_pose = ''
        self.team1_pose = ''
        self.team2_pose = ''

    def field_callback(self, msg):
        self.ball_pose = msg.ball_pose
        self.team1_pose = msg.team1_pose
        self.team2_pose = msg.team2_pose

    def take_action(self, _throttle, _steer):
        msg = CarAction()
        msg.throttle = _throttle
        msg.steer = _steer
        self.publisher.publish(msg)


