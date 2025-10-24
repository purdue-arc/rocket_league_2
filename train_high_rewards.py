#!/usr/bin/env python3
"""
High-Reward Training Script for Rocket League
This script uses optimized parameters to achieve higher reward scores.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

def train_high_rewards():
    """Train with high-reward configuration."""
    print("🚀 High-Reward Rocket League Training")
    print("=" * 50)
    
    # Create high-reward environment
    env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=500,  # Shorter episodes for faster learning
        reward_scale=2.0  # Double all rewards
    )
    
    # Wrap with monitor
    env = Monitor(env, "./high_reward_logs")
    
    # Create evaluation environment
    eval_env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=500,
        reward_scale=2.0
    )
    eval_env = Monitor(eval_env, "./high_reward_eval_logs")
    
    # Create PPO model with optimized parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Higher learning rate
        n_steps=1024,  # More steps per update
        batch_size=128,  # Larger batch size
        n_epochs=20,  # More epochs per update
        gamma=0.99,  # High discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./high_reward_tensorboard"
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./high_reward_best_model",
        log_path="./high_reward_eval_logs",
        eval_freq=5000,  # Evaluate every 5000 steps
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    print("Starting high-reward training...")
    print("Expected reward improvements:")
    print("- Base success: +1000 → +2000 (with 2x scale)")
    print("- Good approach: +500 → +1000 (with 2x scale)")
    print("- Distance rewards: +20x → +40x (with 2x scale)")
    print("- Progressive rewards: +20 → +40 (with 2x scale)")
    
    # Train the model
    model.learn(
        total_timesteps=100000,  # 100k timesteps
        callback=eval_callback,
        tb_log_name="high_reward_rocket_league"
    )
    
    # Save final model
    model.save("./high_reward_final_model")
    print("✅ High-reward model saved!")
    
    # Test the model
    print("\n🧪 Testing high-reward model...")
    test_episodes = 5
    total_rewards = []
    
    for episode in range(test_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    print(f"\n📊 Results:")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Standard deviation: {np.std(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    print("\n🎉 High-reward training completed!")
    print("Check TensorBoard: tensorboard --logdir ./high_reward_tensorboard")

if __name__ == "__main__":
    # Create directories
    os.makedirs("./high_reward_logs", exist_ok=True)
    os.makedirs("./high_reward_eval_logs", exist_ok=True)
    os.makedirs("./high_reward_tensorboard", exist_ok=True)
    
    train_high_rewards()
