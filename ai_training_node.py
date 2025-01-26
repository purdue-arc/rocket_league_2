import rclpy
from rclpy.node import Node
from rktl_interfaces.msg import Field
from rktl_interfaces.msg import CarAction
from stable_baselines3 import PPO, A2C
from rktl_autonomy.envs import TouchBall1


class AiTraining(Node):
    def __init__(self):
        super().__init__('AI_Training')
        self.publisher = self.create_publisher(CarAction, 'car_output', 1)
        self.subscriber = self.create_subscription(Field, 'field_input', self.received_field_data, 1)
        env = TouchBall1()
        self.model = A2C("MultiInputPolicy", env, verbose=1, device="cuda")
        self.model.learn(100000)


    def received_field_data(self, field):
        self.field = field

    def callback_get_obs(self):
        return self.field

    def callback_take_action(self, action):
        # code to publish car action to Car / simulator team
        pass

def main(args=None):
    rclpy.init(args=args)
    node = AiTraining()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()