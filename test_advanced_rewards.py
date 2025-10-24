#!/usr/bin/env python3
"""
Test script for the advanced Rocket League reward system
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

def test_advanced_reward_system():
    """Test the new advanced reward system."""
    print("🎯 Testing Advanced Rocket League Reward System")
    print("=" * 60)
    
    # Create environment
    env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=100,
        reward_scale=1.0
    )
    
    print("✅ Environment created with advanced reward system")
    
    # Test different scenarios
    test_scenarios = [
        ("Basic Movement", [2, 2, 2, 0, 2, 2, 2]),  # Forward movement
        ("Turning", [0, 0, 0, 1, 1, 1, 2]),  # Turn then move
        ("Mixed Strategy", [0, 2, 1, 2, 0, 2, 2]),  # Mixed actions
    ]
    
    for scenario_name, actions in test_scenarios:
        print(f"\n🧪 Testing Scenario: {scenario_name}")
        
        obs, info = env.reset()
        total_reward = 0
        episode_steps = 0
        
        print(f"  Initial position: {env._agent_location}")
        print(f"  Ball position: {env._ball_location}")
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            print(f"  Step {i+1}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
            
            if terminated:
                print(f"  🎉 SUCCESS! Reached the ball in {episode_steps} steps!")
                break
            elif truncated:
                print(f"  ⏰ Episode ended after {episode_steps} steps")
                break
        
        print(f"  Final Reward: {total_reward:.2f}")
        print(f"  Episode Length: {episode_steps}")
    
    env.close()
    print("\n✅ Advanced reward system test completed!")

def test_reward_components():
    """Test individual reward components."""
    print("\n🔍 Testing Reward Components")
    print("=" * 40)
    
    env = RocketLeagueSB3Env(max_episode_steps=50)
    
    # Test offensive rewards
    print("\n📈 Testing Offensive Rewards:")
    obs, _ = env.reset()
    
    # Move toward ball
    for _ in range(5):
        obs, reward, terminated, truncated, _ = env.step(2)  # Forward
        print(f"  Forward step: Reward = {reward:.2f}")
        if terminated or truncated:
            break
    
    # Test defensive positioning
    print("\n🛡️ Testing Defensive Rewards:")
    obs, _ = env.reset()
    
    # Position near goal
    env._agent_location = np.array([50, 150])  # Near goal
    env._ball_location = np.array([100, 150])  # Ball approaching
    
    obs, reward, terminated, truncated, _ = env.step(0)  # Turn
    print(f"  Defensive positioning: Reward = {reward:.2f}")
    
    # Test midfield positioning
    print("\n⚽ Testing Midfield Rewards:")
    obs, _ = env.reset()
    
    # Position in midfield
    env._agent_location = np.array([150, 200])  # Midfield
    env._ball_location = np.array([200, 150])  # Ball in center
    
    obs, reward, terminated, truncated, _ = env.step(2)  # Forward
    print(f"  Midfield positioning: Reward = {reward:.2f}")
    
    env.close()
    print("\n✅ Reward component testing completed!")

def test_reward_scaling():
    """Test different reward scales."""
    print("\n⚖️ Testing Reward Scaling")
    print("=" * 30)
    
    scales = [0.5, 1.0, 2.0, 5.0]
    
    for scale in scales:
        env = RocketLeagueSB3Env(reward_scale=scale, max_episode_steps=20)
        obs, _ = env.reset()
        
        total_reward = 0
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(2)  # Forward
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"  Scale {scale}: Total Reward = {total_reward:.2f}")
        env.close()

def demonstrate_reward_breakdown():
    """Demonstrate how rewards are calculated."""
    print("\n📊 Reward System Breakdown")
    print("=" * 40)
    
    print("""
🎯 OFFENSIVE PLAY REWARDS:
  • Ball Control & Attacking:
    - Base success: +100 points
    - Good approach angle (<30°): +50 points
    - Decent approach angle (<60°): +25 points
    - Poor approach: +10 points
    - Good approach speed: +5x distance improvement
    - Standard approach: +2x distance improvement
    - Good angle to ball (<45°): +2 points
    - Decent angle (<90°): +1 point
    - Poor positioning penalty: -5 points

🛡️ DEFENSIVE PLAY REWARDS:
  • Goal Defense:
    - Good defensive positioning: +10 points
    - Between ball and goal: +15 points
  • Clearance:
    - Successful clearance: +20 points
    - Partial clearance: +5 points
  • Back Positioning:
    - Out of position penalty: -10 points
    - Good defensive position: +5 points

🤝 TEAM COORDINATION REWARDS:
  • Field Position:
    - Good midfield position: +3 points
    - Over-chasing penalty: -2 points
    - Good spacing: +2 points

⚽ POSITIONAL AWARENESS REWARDS:
  • Midfield Play:
    - Good midfield position: +2 points
    - Too far from center: -3 points
  • Time Management:
    - Late episode urgency: +5 points
    - Late episode penalty: -2 points
  • Field Coverage:
    - Good field coverage: +1 point

📊 EFFICIENCY:
  • Step penalty: -0.05 points per step
    """)

if __name__ == "__main__":
    print("🚀 Advanced Rocket League Reward System Test")
    print("=" * 60)
    
    # Run tests
    test_advanced_reward_system()
    test_reward_components()
    test_reward_scaling()
    demonstrate_reward_breakdown()
    
    print("\n🎉 All tests completed!")
    print("\nThe new reward system includes:")
    print("✅ Offensive play rewards (ball control, attacking)")
    print("✅ Defensive play rewards (goal defense, clearance)")
    print("✅ Team coordination rewards (positioning, spacing)")
    print("✅ Positional awareness rewards (midfield, time management)")
    print("✅ Comprehensive reward scaling")
    
    print("\n🚀 Ready for advanced training!")
    print("Run: python examples/quick_start_training.py")
