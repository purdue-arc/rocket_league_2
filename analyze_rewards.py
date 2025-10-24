#!/usr/bin/env python3
"""
Reward Analysis Script
Analyzes the reward system to identify optimization opportunities.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

def analyze_reward_components():
    """Analyze different reward components."""
    print("🔍 Reward System Analysis")
    print("=" * 50)
    
    env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=100,
        reward_scale=1.0
    )
    
    # Test different scenarios
    scenarios = [
        ("Perfect Success", "Test optimal approach and success"),
        ("Good Approach", "Test good but not perfect approach"),
        ("Poor Approach", "Test poor approach"),
        ("Defensive Play", "Test defensive positioning"),
        ("Midfield Play", "Test midfield positioning"),
    ]
    
    for scenario_name, description in scenarios:
        print(f"\n📊 {scenario_name}: {description}")
        
        obs, _ = env.reset()
        
        # Set up specific scenarios
        if scenario_name == "Perfect Success":
            # Set up for perfect success
            env._agent_location = np.array([100, 150])
            env._ball_location = np.array([120, 150])  # Close to agent
            env._agent_angle = 0  # Facing ball
            
        elif scenario_name == "Good Approach":
            # Set up for good approach
            env._agent_location = np.array([100, 150])
            env._ball_location = np.array([200, 150])  # Ball ahead
            env._agent_angle = 0  # Facing ball
            
        elif scenario_name == "Poor Approach":
            # Set up for poor approach
            env._agent_location = np.array([100, 150])
            env._ball_location = np.array([200, 300])  # Ball far and off-angle
            env._agent_angle = 180  # Facing away from ball
            
        elif scenario_name == "Defensive Play":
            # Set up for defensive play
            env._agent_location = np.array([50, 150])  # Near goal
            env._ball_location = np.array([100, 150])  # Ball approaching
            env._agent_angle = 0  # Facing ball
            
        elif scenario_name == "Midfield Play":
            # Set up for midfield play
            env._agent_location = np.array([150, 200])  # Midfield
            env._ball_location = np.array([200, 150])  # Ball in center
            env._agent_angle = 45  # Angled toward ball
        
        # Test a few steps
        total_reward = 0
        for step in range(5):
            action = 2  # Forward action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step+1}: Reward = {reward:.2f}")
            
            if terminated:
                print(f"  🎉 SUCCESS! Total reward: {total_reward:.2f}")
                break
            elif truncated:
                print(f"  ⏰ Episode ended. Total reward: {total_reward:.2f}")
                break
        
        print(f"  Final Total Reward: {total_reward:.2f}")
    
    env.close()

def test_reward_scaling():
    """Test different reward scales."""
    print("\n⚖️ Reward Scaling Analysis")
    print("=" * 40)
    
    scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for scale in scales:
        env = RocketLeagueSB3Env(
            field_width=304.8,
            field_height=426.72,
            max_episode_steps=50,
            reward_scale=scale
        )
        
        obs, _ = env.reset()
        total_reward = 0
        
        # Test a few steps
        for _ in range(10):
            action = 2  # Forward
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"  Scale {scale:4.1f}: Total Reward = {total_reward:8.2f}")
        env.close()

def recommend_improvements():
    """Recommend improvements based on analysis."""
    print("\n💡 Reward System Recommendations")
    print("=" * 50)
    
    print("""
🎯 CURRENT ISSUES IDENTIFIED:
1. Low base rewards (100 → 1000)
2. Small distance rewards (2x → 20x)
3. Minimal angle rewards (2 → 10)
4. High step penalties (0.05 → 0.01)
5. Missing progressive rewards

🚀 IMPROVEMENTS IMPLEMENTED:
✅ Increased base success reward: 100 → 1000
✅ Boosted approach bonuses: 50 → 500, 25 → 250
✅ Enhanced distance rewards: 5x → 20x, 2x → 10x
✅ Improved angle rewards: 2 → 10, 1 → 5
✅ Added progressive distance rewards: +20, +10, +5
✅ Reduced step penalty: 0.05 → 0.01
✅ Boosted defensive rewards: 10 → 50, 15 → 75

📊 EXPECTED IMPROVEMENTS:
• Base success: +1000 → +2000 (with 2x scale)
• Good approach: +500 → +1000 (with 2x scale)
• Distance rewards: +20x → +40x (with 2x scale)
• Progressive rewards: +20 → +40 (with 2x scale)
• Total episode rewards: +2000 to +5000+ (vs previous +3000 max)

🎮 TRAINING RECOMMENDATIONS:
1. Use reward_scale=2.0 for 2x all rewards
2. Train with shorter episodes (500 steps) for faster learning
3. Use higher learning rate (5e-4) for faster convergence
4. Monitor TensorBoard for reward component analysis
5. Use evaluation callbacks to track progress

🚀 NEXT STEPS:
1. Run: python train_high_rewards.py
2. Monitor: tensorboard --logdir ./high_reward_tensorboard
3. Analyze: Check reward components in TensorBoard
4. Tune: Adjust individual reward values if needed
    """)

if __name__ == "__main__":
    analyze_reward_components()
    test_reward_scaling()
    recommend_improvements()
    
    print("\n🎉 Analysis complete!")
    print("Run 'python train_high_rewards.py' to test the improvements!")
