import numpy as np
import rclpy
from rclpy.node import Node

from rktl_interfaces.msg import CarAction, Field


class ROS2Agent(Node):
    def __init__(self, env_id, model):
        # Create a node with a unique name for this environment instance
        super().__init__(f'ros2_agent_{env_id}')
        self.env_id = env_id
        self.model = model  # Your trained or currently-training model
        
        # Unique topics per instance
        self.state_topic = f'/simulator/{env_id}/state'
        self.action_topic = f'/simulator/{env_id}/action'
        
        # Create subscriber for state messages
        self.create_subscription(
            Field, 
            self.state_topic, 
            self.state_callback, 
            10
        )
        # Create publisher for action messages
        self.action_publisher = self.create_publisher(Field, self.action_topic, 10)
        
    def state_callback(self, msg):
        # Parse the incoming state message.
        # Assume msg.ball and msg.car each have: id, x, y, angle_degrees.
        observation = np.array([
            msg.ball.id, msg.ball.x, msg.ball.y, msg.ball.angle_degrees,
            msg.car.id,  msg.car.x,  msg.car.y,  msg.car.angle_degrees,
        ], dtype=np.float64)
        
        # Predict an action using your AI model.
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Package the predicted action into a ROS2 message.
        action_msg = CarAction()
        # Assuming your action message has fields 'x' and 'y' for the two floats.
        action_msg.id = 0
        action_msg.throttle = float(action[0])
        action_msg.steer = float(action[1])
        
        # Publish the action back to the simulator.
        self.action_publisher.publish(action_msg)
        self.get_logger().info(f"Env {self.env_id}: Received state and published action {action}")

    def run(self):
        # Run a simple spin loop so that callbacks are processed.
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

# Example usage:
def main():
    rclpy.init()
    # Assume 'model' is your Stable Baselines 3 model, loaded or being trained.
    # For a vectorized environment, you’d instantiate multiple such nodes,
    # each with its unique env_id and corresponding ROS2 topics.
    env_id = 0
    model = ...  # Load or create your model here.
    agent = ROS2Agent(env_id, model)
    
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
