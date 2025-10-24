#!/usr/bin/env python3
"""
Test script to verify the SB3 integration fix
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

def test_environment():
    """Test the environment with numpy array actions."""
    print("🧪 Testing Rocket League SB3 Environment")
    print("=" * 50)
    
    # Create environment
    env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=100
    )
    
    print("✅ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful - Observation shape: {obs.shape}")
    
    # Test step with different action types
    test_actions = [
        0,  # Integer
        1,  # Integer
        2,  # Integer
        np.array([0]),  # Numpy array
        np.array([1]),  # Numpy array
        np.array([2]),  # Numpy array
    ]
    
    print("\nTesting different action types:")
    for i, action in enumerate(test_actions):
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ Action {i+1} ({type(action).__name__}): Reward = {reward:.2f}")
            
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
                
        except Exception as e:
            print(f"❌ Action {i+1} failed: {e}")
            return False
    
    # Test action distribution
    print(f"\nAction distribution: {env.action_dist}")
    
    # Clean up
    env.close()
    print("✅ Environment closed successfully")
    
    return True

def test_training_step():
    """Test a short training step."""
    print("\n🧪 Testing Training Step")
    print("=" * 50)
    
    try:
        from stable_baselines3 import PPO
        
        # Create environment
        env = RocketLeagueSB3Env(
            field_width=304.8,
            field_height=426.72,
            max_episode_steps=50
        )
        
        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=0)
        
        print("✅ PPO model created successfully")
        
        # Train for a few steps
        model.learn(total_timesteps=100)
        
        print("✅ Training completed successfully")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"✅ Prediction successful - Action type: {type(action)}, Value: {action}")
        
        # Test step with predicted action
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"✅ Step successful - Reward: {reward:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing SB3 Integration Fix")
    print("=" * 60)
    
    # Test 1: Basic environment functionality
    test1_passed = test_environment()
    
    # Test 2: Training integration
    test2_passed = test_training_step()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! The fix is working correctly.")
        print("\nYou can now run:")
        print("python examples/quick_start_training.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
