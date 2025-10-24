#!/usr/bin/env python3
"""
Quick Start Training Example for Rocket League with Stable Baselines 3
This example shows how to quickly get started with training an RL agent.
"""

import os
import sys
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

def quick_training_example():
    """Quick example of training a PPO agent."""
    
    print("🚀 Starting Rocket League RL Training Example")
    print("=" * 50)
    
    # Create environment
    print("Creating environment...")
    env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=500,  # Shorter episodes for quick training
        reward_scale=1.0
    )
    
    # Wrap with monitor for logging
    env = Monitor(env, "./quick_start_logs")
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
        verbose=1,
        tensorboard_log="./quick_start_tensorboard"
    )
    
    # Train for a short time
    print("Training model (this will take a few minutes)...")
    model.learn(total_timesteps=10000)
    
    # Save the model
    model_path = "./quick_start_model"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Test the trained model
    print("Testing trained model...")
    test_episodes = 5
    total_rewards = []
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:  # Limit steps for quick testing
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    print(f"\nAverage reward over {test_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard deviation: {np.std(total_rewards):.2f}")
    
    # Clean up
    env.close()
    
    print("\n✅ Quick training example completed!")
    print("Check the following files:")
    print("- ./quick_start_logs/ - Training logs")
    print("- ./quick_start_tensorboard/ - TensorBoard logs")
    print("- ./quick_start_model.zip - Trained model")

def advanced_training_example():
    """More advanced training example with evaluation."""
    
    print("\n🔬 Advanced Training Example")
    print("=" * 50)
    
    # Create training and evaluation environments
    train_env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=1000,
        reward_scale=1.0
    )
    
    eval_env = RocketLeagueSB3Env(
        field_width=304.8,
        field_height=426.72,
        max_episode_steps=1000,
        reward_scale=1.0
    )
    
    # Wrap environments
    train_env = Monitor(train_env, "./advanced_logs")
    eval_env = Monitor(eval_env, "./advanced_eval_logs")
    
    # Create model with custom parameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./advanced_tensorboard"
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./advanced_best_model",
        log_path="./advanced_eval_logs",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train with evaluation
    print("Training with evaluation...")
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        tb_log_name="advanced_rocket_league"
    )
    
    # Final evaluation
    print("Final evaluation...")
    final_rewards = []
    for episode in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        final_rewards.append(episode_reward)
        print(f"Final Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nFinal Average Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    print("\n✅ Advanced training example completed!")

def hyperparameter_tuning_example():
    """Example of hyperparameter tuning with Optuna."""
    
    try:
        import optuna
        from optuna.integration import SB3Callback
        
        print("\n🎯 Hyperparameter Tuning Example")
        print("=" * 50)
        
        def objective(trial):
            """Objective function for hyperparameter optimization."""
            
            # Sample hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
            n_epochs = trial.suggest_int("n_epochs", 3, 20)
            
            # Create environment
            env = RocketLeagueSB3Env(
                field_width=304.8,
                field_height=426.72,
                max_episode_steps=500,
                reward_scale=1.0
            )
            env = Monitor(env, f"./tuning_logs/trial_{trial.number}")
            
            # Create model with sampled hyperparameters
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                n_epochs=n_epochs,
                verbose=0
            )
            
            # Train for a short time
            model.learn(total_timesteps=5000)
            
            # Evaluate
            eval_rewards = []
            for _ in range(5):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                eval_rewards.append(episode_reward)
            
            env.close()
            return np.mean(eval_rewards)
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        print("Best hyperparameters:")
        print(study.best_params)
        print(f"Best value: {study.best_value:.2f}")
        
    except ImportError:
        print("Optuna not installed. Install with: pip install optuna")
        print("Skipping hyperparameter tuning example.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./quick_start_logs", exist_ok=True)
    os.makedirs("./quick_start_tensorboard", exist_ok=True)
    os.makedirs("./advanced_logs", exist_ok=True)
    os.makedirs("./advanced_eval_logs", exist_ok=True)
    os.makedirs("./tuning_logs", exist_ok=True)
    
    # Run examples
    quick_training_example()
    advanced_training_example()
    hyperparameter_tuning_example()
    
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. View training logs with TensorBoard: tensorboard --logdir ./quick_start_tensorboard")
    print("2. Try different algorithms: DQN, A2C, SAC")
    print("3. Experiment with different reward functions")
    print("4. Integrate with the full ROS 2 simulation")
