# Stable Baselines 3 Integration for Rocket League

This guide explains how to integrate Stable Baselines 3 (SB3) with your Rocket League simulator for reinforcement learning training.

## Overview

The integration provides:
- **SB3-compatible environment**: Optimized for RL training
- **Multiple algorithms**: PPO, DQN, A2C, SAC support
- **ROS 2 integration**: Train with real simulation data
- **Comprehensive training scripts**: Ready-to-use examples
- **Evaluation and monitoring**: TensorBoard, logging, metrics

## Quick Start

### 1. Install Dependencies

```bash
# Install SB3 and dependencies
pip install stable-baselines3[extra] tensorboard matplotlib

# Or install all requirements
pip install -r requirements.txt
```

### 2. Basic Training Example

```python
from stable_baselines3 import PPO
from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

# Create environment
env = RocketLeagueSB3Env(
    field_width=304.8,
    field_height=426.72,
    max_episode_steps=1000
)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100000)

# Save model
model.save("rocket_league_agent")
```

### 3. Run Examples

```bash
# Quick start training
python examples/quick_start_training.py

# Full training with evaluation
python rktl_autonomy/train_sb3.py --algorithm PPO --timesteps 100000

# ROS integration training
python examples/ros_integration_example.py
```

## Environment Details

### Observation Space
The environment provides a normalized 7-dimensional observation:
- `[0:2]`: Agent position (normalized to [-1, 1])
- `[2]`: Agent angle (normalized to [-1, 1])
- `[3:5]`: Ball position (normalized to [-1, 1])
- `[5]`: Distance to ball (normalized to [0, 1])
- `[6]`: Angle to ball (normalized to [-1, 1])

### Action Space
Discrete actions:
- `0`: Turn right
- `1`: Turn left  
- `2`: Move forward

### Reward Function
- **+100**: Reaching the ball
- **+2.0**: Getting closer to ball (per unit distance)
- **+0.5**: Turning toward ball
- **-0.1**: Per step penalty (encourage efficiency)
- **-1.0**: Getting further from ball
- **-2.0**: Turning away from ball

## Training Scripts

### 1. Basic Training (`train_sb3.py`)

Full-featured training script with:
- Multiple algorithm support (PPO, DQN, A2C, SAC)
- Evaluation callbacks
- TensorBoard logging
- Model saving/loading
- Command-line interface

```bash
# Train with PPO
python rktl_autonomy/train_sb3.py --algorithm PPO --timesteps 100000

# Train with DQN
python rktl_autonomy/train_sb3.py --algorithm DQN --timesteps 50000

# Load and continue training
python rktl_autonomy/train_sb3.py --load-model ./models/PPO_final_model
```

### 2. Quick Start (`examples/quick_start_training.py`)

Simple examples for getting started:
- Basic training loop
- Advanced training with evaluation
- Hyperparameter tuning with Optuna

### 3. ROS Integration (`examples/ros_integration_example.py`)

Train with the actual ROS 2 simulation:
- Real-time data from simulator
- Action publishing to ROS topics
- Integration with existing ROS nodes

## Algorithm Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **PPO** | General purpose | Stable, sample efficient | Slower training |
| **DQN** | Discrete actions | Good exploration | Can be unstable |
| **A2C** | Quick training | Fast, simple | Less stable than PPO |
| **SAC** | Continuous control | Sample efficient | More complex |

## Monitoring and Evaluation

### TensorBoard
```bash
# View training progress
tensorboard --logdir ./tensorboard_logs

# View specific run
tensorboard --logdir ./tensorboard_logs/PPO_rocket_league_1
```

### Evaluation Metrics
- **Episode Reward**: Total reward per episode
- **Episode Length**: Steps per episode
- **Success Rate**: Percentage of episodes reaching the ball
- **Action Distribution**: Balance of actions taken

## Advanced Features

### 1. Custom Reward Functions

Modify the reward function in `RocketLeagueSB3Env._calculate_reward()`:

```python
def _calculate_reward(self, prev_distance, current_distance, 
                    angle_diff_before, angle_diff_after, terminated):
    reward = 0.0
    
    # Custom reward logic
    if terminated:
        reward += 1000.0  # Higher reward for success
    
    # Distance-based reward
    distance_improvement = prev_distance - current_distance
    reward += distance_improvement * 5.0  # Higher multiplier
    
    return reward * self.reward_scale
```

### 2. Environment Wrappers

Use SB3 environment wrappers for additional functionality:

```python
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

# Vectorized environment
env = make_vec_env(lambda: RocketLeagueSB3Env(), n_envs=4)

# Normalize observations
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Monitor environment
env = VecMonitor(env)
```

### 3. Hyperparameter Tuning

Use Optuna for automatic hyperparameter optimization:

```python
import optuna
from optuna.integration import SB3Callback

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size)
    model.learn(total_timesteps=10000)
    
    # Evaluate and return score
    return evaluate_model(model, env)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## ROS 2 Integration

### Prerequisites
1. ROS 2 Jazzy installed
2. Workspace built: `colcon build`
3. Simulator running: `ros2 run rktl_simulator simulator.py`

### Training with ROS
```python
from examples.ros_integration_example import ROSRocketLeagueEnv

# Create ROS-integrated environment
env = ROSRocketLeagueEnv(
    field_width=304.8,
    field_height=426.72,
    max_episode_steps=1000
)

# Train with real simulation data
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000)
```

### Topics Used
- **Subscribed**: `/FieldState` - Receives game state
- **Published**: `/CarAction` - Sends control commands

## Troubleshooting

### Common Issues

1. **Environment not resetting properly**
   ```python
   # Ensure proper reset
   obs, info = env.reset()
   ```

2. **ROS connection issues**
   ```bash
   # Check ROS topics
   ros2 topic list
   ros2 topic echo /FieldState
   ```

3. **Training not converging**
   - Try different learning rates
   - Adjust reward function
   - Increase training time
   - Use different algorithm

### Performance Tips

1. **Faster Training**
   - Use vectorized environments
   - Reduce episode length for quick iterations
   - Use GPU if available

2. **Better Convergence**
   - Normalize observations
   - Tune reward function
   - Use curriculum learning
   - Add domain randomization

## File Structure

```
rktl_autonomy/
├── training_env/
│   └── envs/
│       └── rocket_league_sb3.py    # SB3 environment
├── train_sb3.py                    # Main training script
└── examples/
    ├── quick_start_training.py     # Quick start examples
    └── ros_integration_example.py  # ROS integration
```

## Next Steps

1. **Experiment with algorithms**: Try DQN, A2C, SAC
2. **Tune hyperparameters**: Use Optuna for optimization
3. **Customize rewards**: Modify reward function for your goals
4. **Scale training**: Use multiple environments
5. **Deploy**: Integrate trained model with ROS system

## Resources

- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [ROS 2 Documentation](https://docs.ros.org/en/jazzy/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
