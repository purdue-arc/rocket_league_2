import math

import gymnasium
import numpy as np
import rclpy
from gymnasium import spaces

from rktl_interfaces.msg import CarAction, Field, Pose


class CustomROS2Env(gymnasium.Env):
    REWARD_MAGNITUDE_BIAS = 0.75
    REWARD_ANGLE_BIAS = 0.25
    def __init__(self, envID):
        super().__init__()
        self.numInputs = 8
        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(8,), dtype=np.float64)
        # Define the action space (2 continuous outputs)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       shape=(2,), dtype=np.float64)
        
        self.currentObs = np.zeros((self.numInputs, 2), dtype=np.float64)
        self.envID = envID
        self._setup_ros2_nodes()

    def _setup_ros2_nodes(self, envID):
        rclpy.init()
        self.node = rclpy.create_node(f"custom_env_node_{self.envID}")
        self.publisher = self.node.create_publisher(CarAction, f"car_action_{self.envID}", 10)
        self.subscriber = self.node.create_subscription(Field, f"field_state_{self.envID}", self._state_callback, 10)
        self.gotMessage = False

    def _state_callback(self, msg):
        # Process the incoming ROS2 message to update self.current_obs
        # self.current_obs = ...
        ballPose = msg.ball_pose
        ball = [ballPose.id, ballPose.x, ballPose.y, ballPose.angle_degrees]
        carPose = msg.team1_poses[0]
        car = [carPose.id, carPose.x, carPose.y, carPose.angle_degrees]
        self.currentObs = np.array(ball + car, dtype=np.float64)
        self.gotMessage = True

    def reset(self):
        # Publish a reset command to the simulator if needed
        # and wait for the initial state message (consider using a blocking mechanism or a short timeout loop)
        self.currentObs = np.zeros((self.numInputs, 2), dtype=np.float64)  # Replace with actual reset logic
        return self.currentObs

    def step(self, action):
        # Publish the action to the simulator via your ROS2 publisher.
        # For example, convert the action to your custom ROS2 message and publish it.
        
        publishAction = CarAction()
        publishAction.id = 0
        publishAction.throttle = action[0]
        publishAction.steer = action[1]
        self.publisher.publish(publishAction)
        
        while (not self.gotMessage):
            pass
        self.gotMessage = True
        
        # Loads the recieved observation
        nextObs = self.currentObs
        
        # Compute reward based on the simulator's feedback
        ballPosition = [self.currentObs[1], self.currentObs[2]]
        carPosition = [self.currentObs[5], self.currentObs[6], self.currentObs[7]]
        difference = [ballPosition[0] - carPosition[0], ballPosition[1] - carPosition[1]]
        magnitude = -((difference[0] ** 2) + (difference[1] ** 2)) ** 0.5
        
        # Angle reward
        angle = math.atan(difference[1]/difference[0])
        angleDifference = carPosition[3] - angle
        angleReward = -np.interp(abs(angleDifference), [0, 2 * math.pi], [0, 100])
        
        reward = (magnitude * CustomROS2Env.REWARD_MAGNITUDE_BIAS +
            angleReward * CustomROS2Env.REWARD_ANGLE_BIAS)
        
        # Determine whether the episode is done.
        done = False  # Replace with your termination condition.
        
        # Additional information can be added to the info dictionary.
        info = {}
        
        return nextObs, reward, done, info

    def render(self, mode='human'):
        # Optional: Implement visualization if necessary.
        pass

    def close(self):
        # Clean up ROS2 nodes and any other resources.
        # For example: self.node.destroy_node() and rclpy.shutdown()
        self.node.destroy_node()
