import gymnasium
import numpy as np
from gymnasium import spaces


class CustomROS2Env(gymnasium.Env):
    def __init__(self, envID):
        super().__init__()
        self.num_points = 2  # Adjust this to your requirements
        # Define the observation space (e.g., 5 (double, double) points)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(8,), dtype=np.float64)
        # Define the action space (2 continuous outputs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        
        self.current_obs = np.zeros((self.num_points, 2), dtype=np.float64)
        self.envID = envID
        self._setup_ros2_nodes()

    def _setup_ros2_nodes(self, envID):
        import rclpy
        from rktl_interfaces.msg import CarAction, Field, Pose
        rclpy.init()
        self.node = rclpy.create_node(f"custom_env_node_{self.envID}")
        self.publisher = self.node.create_publisher(CarAction, f"car_action_{self.envID}", 10)
        self.subscriber = self.node.create_subscription(Field, f"field_state_{self.envID}", self._state_callback, 10)

    def _state_callback(self, msg):
        # Process the incoming ROS2 message to update self.current_obs
        # For example, parse custom message fields and convert to a NumPy array.
        # self.current_obs = ...
        ballPose = msg.ball_pose
        ball = [ballPose.id, ballPose.x, ballPose.y, ballPose.angle_degrees]
        carPose = msg.team1_poses[0]
        car = [carPose.id, carPose.x, carPose.y, carPose.angle_degrees]
        self.current_obs = np.array(ball + car, dtype=np.float64)
        

    def reset(self):
        # Publish a reset command to the simulator if needed
        # and wait for the initial state message (consider using a blocking mechanism or a short timeout loop)
        self.current_obs = np.zeros((self.num_points, 2), dtype=np.float64)  # Replace with actual reset logic
        return self.current_obs

    def step(self, action):
        # Publish the action to the simulator via your ROS2 publisher.
        # For example, convert the action to your custom ROS2 message and publish it.
        # self.publisher.publish(your_message)
        
        # Optionally, wait or poll until you receive the next state via the subscriber callback.
        # This might involve a blocking wait or checking a shared variable updated in _state_callback.
        next_obs = self.current_obs  # This should be updated with the new state from ROS2.
        
        # Compute reward based on the simulator's feedback (either embedded in the ROS2 message or computed here).
        reward = 0.0  # Replace with your reward logic.
        
        # Determine whether the episode is done.
        done = False  # Replace with your termination condition.
        
        # Additional information can be added to the info dictionary.
        info = {}
        
        return next_obs, reward, done, info

    def render(self, mode='human'):
        # Optional: Implement visualization if necessary.
        pass

    def close(self):
        # Clean up ROS2 nodes and any other resources.
        # For example: self.node.destroy_node() and rclpy.shutdown()
        pass
