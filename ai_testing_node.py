# import rclpy
# from rclpy.node import Node
# from rktl_interfaces.msg import Field
# from rktl_interfaces.msg import CarAction
from stable_baselines3 import PPO, A2C

class AiTesting(Node):
    def __init__(self):
        super().__init__('AI_Testing')
        self.publisher = self.create_publisher(CarAction, 'car_output', 1)
        self.subscriber = self.create_subscription(Field, 'field_input', self.received_field_data, 1)
        self.model = PPO.load(model_path, env=env)

    def received_field_data(self, field):
        action, _state = self.model.predict(field)
        msg = CarAction()
        msg._id = 0
        msg._throttle = action[0]
        msg._steer = action[1]

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AiTesting()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()