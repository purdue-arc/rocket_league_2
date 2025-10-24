#!/usr/bin/env python3
"""
Stable Baselines 3 Training Script for Rocket League Environment
This script demonstrates how to train RL agents using SB3 with the Rocket League environment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import argparse
import json
from datetime import datetime

# Stable Baselines 3 imports
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Local imports
from training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Environment parameters
        self.field_width = 304.8
        self.field_height = 426.72
        self.max_episode_steps = 1000
        self.reward_scale = 1.0
        
        # Training parameters
        self.algorithm = "PPO"  # PPO, DQN, A2C, SAC
        self.total_timesteps = 100000
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.buffer_size = 10000
        
        # Evaluation parameters
        self.eval_freq = 10000
        self.n_eval_episodes = 10
        self.eval_reward_threshold = 50.0
        
        # Logging parameters
        self.log_dir = "./logs"
        self.tensorboard_log = "./tensorboard_logs"
        self.save_freq = 10000
        
        # Model parameters
        self.policy = "MlpPolicy"  # MlpPolicy, CnnPolicy
        self.verbose = 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "field_width": self.field_width,
            "field_height": self.field_height,
            "max_episode_steps": self.max_episode_steps,
            "reward_scale": self.reward_scale,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "eval_reward_threshold": self.eval_reward_threshold,
            "policy": self.policy
        }

def create_environment(config: TrainingConfig, render_mode: Optional[str] = None) -> RocketLeagueSB3Env:
    """Create and configure the training environment."""
    env = RocketLeagueSB3Env(
        field_width=config.field_width,
        field_height=config.field_height,
        render_mode=render_mode,
        max_episode_steps=config.max_episode_steps,
        reward_scale=config.reward_scale
    )
    return env

def create_model(config: TrainingConfig, env) -> Any:
    """Create the RL model based on configuration."""
    
    model_kwargs = {
        "learning_rate": config.learning_rate,
        "verbose": config.verbose,
        "tensorboard_log": config.tensorboard_log
    }
    
    if config.algorithm == "PPO":
        model = PPO(
            config.policy,
            env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log
        )
    elif config.algorithm == "DQN":
        model = DQN(
            config.policy,
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log
        )
    elif config.algorithm == "A2C":
        model = A2C(
            config.policy,
            env,
            learning_rate=config.learning_rate,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log
        )
    elif config.algorithm == "SAC":
        model = SAC(
            config.policy,
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            verbose=config.verbose,
            tensorboard_log=config.tensorboard_log
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    return model

def train_model(config: TrainingConfig, model, env, eval_env) -> Any:
    """Train the model with evaluation callbacks."""
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config.log_dir}/best_model",
        log_path=f"{config.log_dir}/eval_logs",
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    # Optional: Stop training when reward threshold is reached
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=config.eval_reward_threshold,
        verbose=1
    )
    
    # Train the model
    print(f"Starting training with {config.algorithm}...")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback],
        tb_log_name=f"{config.algorithm}_rocket_league"
    )
    
    return model

def evaluate_model(model, env, n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
    """Evaluate the trained model."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }

def plot_training_results(log_dir: str, save_path: Optional[str] = None):
    """Plot training results from logs."""
    try:
        # Load results
        results = load_results(log_dir)
        x, y = ts2xy(results, 'timesteps')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(x, y)
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress - Episode Rewards')
        plt.grid(True)
        
        # Plot moving average
        if len(y) > 10:
            window = min(100, len(y) // 10)
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            plt.plot(x[window-1:], moving_avg, label=f'Moving Average (window={window})', alpha=0.7)
            plt.legend()
        
        # Plot episode lengths
        plt.subplot(2, 1, 2)
        episode_lengths = results['l'].values
        plt.plot(episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {e}")

def save_training_config(config: TrainingConfig, save_path: str):
    """Save training configuration to file."""
    config_dict = config.to_dict()
    config_dict["timestamp"] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training configuration saved to: {save_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for Rocket League")
    parser.add_argument("--algorithm", type=str, default="PPO", 
                       choices=["PPO", "DQN", "A2C", "SAC"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--render", action="store_true",
                       help="Render during evaluation")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to load existing model")
    parser.add_argument("--save-dir", type=str, default="./models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig()
    config.algorithm = args.algorithm
    config.total_timesteps = args.timesteps
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.eval_freq = args.eval_freq
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, "training_config.json")
    save_training_config(config, config_path)
    
    # Create environments
    print("Creating training environment...")
    train_env = create_environment(config)
    train_env = Monitor(train_env, config.log_dir)
    
    print("Creating evaluation environment...")
    eval_env = create_environment(config)
    eval_env = Monitor(eval_env, f"{config.log_dir}/eval")
    
    # Create model
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        if config.algorithm == "PPO":
            model = PPO.load(args.load_model)
        elif config.algorithm == "DQN":
            model = DQN.load(args.load_model)
        elif config.algorithm == "A2C":
            model = A2C.load(args.load_model)
        elif config.algorithm == "SAC":
            model = SAC.load(args.load_model)
        model.set_env(train_env)
    else:
        print("Creating new model...")
        model = create_model(config, train_env)
    
    # Train model
    print("Starting training...")
    trained_model = train_model(config, model, train_env, eval_env)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"{config.algorithm}_final_model")
    trained_model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Evaluate model
    print("Evaluating trained model...")
    eval_results = evaluate_model(trained_model, eval_env, n_episodes=10, render=args.render)
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Mean Episode Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
    
    # Plot results
    plot_path = os.path.join(args.save_dir, "training_plot.png")
    plot_training_results(config.log_dir, plot_path)
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
