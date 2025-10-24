#!/usr/bin/env python3
"""
ROS 2 Integration Example for Rocket League Training
This example shows how to integrate the SB3 training with the ROS 2 simulation.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node
from rktl_interfaces.msg import Field, CarAction, Pose
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

class ROSRocketLeagueEnv(RocketLeagueSB3Env):
    """
    ROS 2 integrated version of the Rocket League environment.
    This environment receives observations from the ROS simulator and sends actions back.
    """
    
    def __init__(self, node_name: str = "rl_agent", **kwargs):
        """Initialize the ROS-integrated environment."""
        super().__init__(**kwargs)
        
        # Initialize ROS 2
        rclpy.init()
        self.node = Node(node_name)
        
        # ROS communication
        self.field_subscription = self.node.create_subscription(
            Field,
            'FieldState',
            self.field_callback,
            10
        )
        
        self.action_publisher = self.node.create_publisher(
            CarAction,
            'CarAction',
            10
        )
        
        # State variables
        self.latest_field_data = None
        self.episode_started = False
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        # Threading for ROS spinning
        self.ros_thread = threading.Thread(target=self._spin_ros)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        
        print("ROS Rocket League Environment initialized")
    
    def _spin_ros(self):
        """Spin ROS in a separate thread."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.001)
    
    def field_callback(self, msg: Field):
        """Callback for receiving field state from simulator."""
        self.latest_field_data = msg
        
        # Extract agent and ball positions
        if msg.team1_poses:  # Assuming agent is team1
            agent_pose = msg.team1_poses[0]
            self._agent_location = np.array([agent_pose.x, agent_pose.y], dtype=np.float32)
            self._agent_angle = agent_pose.angle_degrees
        
        # Extract ball position
        self._ball_location = np.array([msg.ball_pose.x, msg.ball_pose.y], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        
        # Reset episode variables
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_started = True
        
        # Wait for initial field data
        timeout = 5.0  # 5 second timeout
        start_time = time.time()
        while self.latest_field_data is None and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if self.latest_field_data is None:
            raise RuntimeError("Timeout waiting for field data from ROS simulator")
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int):
        """Execute one step in the environment."""
        if not self.episode_started:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.episode_steps += 1
        
        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Convert action to ROS message
        car_action = CarAction()
        car_action.id = 0  # Agent ID
        
        if action == 0:  # TURN_R
            car_action.throttle = 0.0
            car_action.steer = 1.0
        elif action == 1:  # TURN_L
            car_action.throttle = 0.0
            car_action.steer = -1.0
        elif action == 2:  # FORWARD
            car_action.throttle = 1.0
            car_action.steer = 0.0
        
        # Publish action
        self.action_publisher.publish(car_action)
        
        # Wait for next field update (simulate step time)
        time.sleep(0.1)  # 10 FPS
        
        # Calculate reward based on current state
        if self.latest_field_data is not None:
            # Update internal state from ROS data
            if self.latest_field_data.team1_poses:
                agent_pose = self.latest_field_data.team1_poses[0]
                self._agent_location = np.array([agent_pose.x, agent_pose.y], dtype=np.float32)
                self._agent_angle = agent_pose.angle_degrees
            
            self._ball_location = np.array([
                self.latest_field_data.ball_pose.x, 
                self.latest_field_data.ball_pose.y
            ], dtype=np.float32)
        
        # Calculate reward
        reward = self._calculate_ros_reward()
        self.episode_reward += reward
        
        # Check termination
        distance = np.linalg.norm(self._agent_location - self._ball_location)
        terminated = distance <= self.close_enough_radius
        truncated = self.episode_steps >= self.max_episode_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_ros_reward(self) -> float:
        """Calculate reward based on ROS field data."""
        if self.latest_field_data is None:
            return 0.0
        
        # Simple reward based on distance to ball
        distance = np.linalg.norm(self._agent_location - self._ball_location)
        
        # Reward for getting closer to ball
        if distance <= self.close_enough_radius:
            return 100.0  # Large reward for reaching ball
        else:
            # Small reward for being closer to ball
            return max(0.0, 10.0 - distance * 0.1)
    
    def close(self):
        """Close the environment and clean up ROS resources."""
        super().close()
        if hasattr(self, 'node'):
            self.node.destroy_node()
        rclpy.shutdown()

def train_with_ros_simulation():
    """Train an agent using the ROS 2 simulation."""
    
    print("🤖 Training with ROS 2 Simulation")
    print("=" * 50)
    print("Make sure the ROS simulator is running:")
    print("ros2 run rktl_simulator simulator.py")
    print()
    
    # Create ROS-integrated environment
    env = ROSRocketLeagueEnv(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=1000,
        reward_scale=1.0
    )
    
    # Wrap with monitor
    env = Monitor(env, "./ros_training_logs")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
        verbose=1,
        tensorboard_log="./ros_tensorboard"
    )
    
    print("Starting training with ROS simulation...")
    print("This will interact with the running ROS simulator.")
    
    # Train for a short time
    model.learn(total_timesteps=10000)
    
    # Save model
    model.save("./ros_trained_model")
    print("Model saved to: ./ros_trained_model.zip")
    
    # Test the model
    print("Testing trained model...")
    test_episodes = 3
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"Episode {episode + 1}:")
        
        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
            
            if step % 20 == 0:
                print(f"  Step {step}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")
        
        print(f"  Final Reward: {episode_reward:.2f}, Steps: {step}")
    
    env.close()
    print("✅ ROS training completed!")

def load_and_test_model():
    """Load a trained model and test it with ROS simulation."""
    
    print("🧪 Testing Trained Model with ROS")
    print("=" * 50)
    
    # Load model
    try:
        model = PPO.load("./ros_trained_model")
        print("Model loaded successfully")
    except FileNotFoundError:
        print("No trained model found. Run training first.")
        return
    
    # Create environment
    env = ROSRocketLeagueEnv(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=1000,
        reward_scale=1.0
    )
    
    print("Testing model with ROS simulation...")
    print("Make sure the ROS simulator is running!")
    
    # Test episodes
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")
        
        print(f"  Final Reward: {episode_reward:.2f}, Steps: {step}")
    
    env.close()
    print("✅ Model testing completed!")

def main():
    """Main function for ROS integration example."""
    
    print("🚀 Rocket League ROS 2 Integration Example")
    print("=" * 60)
    print()
    print("This example shows how to integrate SB3 training with ROS 2 simulation.")
    print()
    print("Prerequisites:")
    print("1. ROS 2 Jazzy installed and sourced")
    print("2. Rocket League packages built: colcon build")
    print("3. ROS simulator running: ros2 run rktl_simulator simulator.py")
    print()
    
    # Create directories
    os.makedirs("./ros_training_logs", exist_ok=True)
    os.makedirs("./ros_tensorboard", exist_ok=True)
    
    try:
        # Check if ROS is available
        import rclpy
        print("✅ ROS 2 is available")
        
        # Run training
        train_with_ros_simulation()
        
        # Test the model
        load_and_test_model()
        
    except ImportError as e:
        print(f"❌ ROS 2 not available: {e}")
        print("Please install ROS 2 Jazzy and source the workspace")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the ROS simulator is running")

if __name__ == "__main__":
    main()
