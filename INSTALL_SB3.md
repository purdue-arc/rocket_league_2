# Stable Baselines 3 Installation and Setup Guide

This guide will help you set up Stable Baselines 3 integration with your Rocket League project.

## Prerequisites

- Python 3.8 or higher
- ROS 2 Jazzy (for ROS integration)
- Git

## Installation Steps

### 1. Install Python Dependencies

```bash
# Navigate to your project directory
cd /Users/hhatami/rocket_league_2

# Install core dependencies
pip install stable-baselines3[extra]>=2.0.0
pip install tensorboard>=2.8.0
pip install matplotlib>=3.5.0
pip install gymnasium>=0.28.0

# Install all project dependencies
pip install -r requirements.txt
```

### 2. Install ROS 2 Dependencies (Optional)

If you want to use ROS 2 integration:

```bash
# Source ROS 2 (adjust path as needed)
source /opt/ros/jazzy/setup.bash

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

### 3. Verify Installation

```bash
# Test SB3 installation
python -c "import stable_baselines3; print('SB3 installed successfully')"

# Test environment
python -c "from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env; print('Environment imported successfully')"
```

## Quick Start

### 1. Basic Training

```bash
# Run quick start example
python examples/quick_start_training.py

# Or use the main training script
python rktl_autonomy/train_sb3.py --algorithm PPO --timesteps 10000
```

### 2. ROS Integration Training

```bash
# Terminal 1: Start ROS simulator
ros2 run rktl_simulator simulator.py

# Terminal 2: Run ROS training
python examples/ros_integration_example.py
```

### 3. View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir ./tensorboard_logs

# Open browser to http://localhost:6006
```

## Usage Examples

### Example 1: Basic PPO Training

```python
from stable_baselines3 import PPO
from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

# Create environment
env = RocketLeagueSB3Env(
    field_width=304.8,
    field_height=426.72,
    max_episode_steps=1000
)

# Create and train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save model
model.save("my_rocket_league_agent")
```

### Example 2: Training with Evaluation

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rktl_autonomy.training_env.envs.rocket_league_sb3 import RocketLeagueSB3Env

# Create training and evaluation environments
train_env = RocketLeagueSB3Env(max_episode_steps=1000)
eval_env = RocketLeagueSB3Env(max_episode_steps=1000)

# Create model
model = PPO("MlpPolicy", train_env, verbose=1)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model",
    eval_freq=10000,
    n_eval_episodes=10
)

# Train with evaluation
model.learn(total_timesteps=100000, callback=eval_callback)
```

### Example 3: ROS Integration

```python
from examples.ros_integration_example import ROSRocketLeagueEnv
from stable_baselines3 import PPO

# Create ROS-integrated environment
env = ROSRocketLeagueEnv(
    field_width=304.8,
    field_height=426.72,
    max_episode_steps=1000
)

# Train with real simulation data
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Test the trained model
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
```

## Command Line Usage

### Training Script Options

```bash
# Basic training
python rktl_autonomy/train_sb3.py --algorithm PPO --timesteps 100000

# Different algorithms
python rktl_autonomy/train_sb3.py --algorithm DQN --timesteps 50000
python rktl_autonomy/train_sb3.py --algorithm A2C --timesteps 75000
python rktl_autonomy/train_sb3.py --algorithm SAC --timesteps 100000

# Custom parameters
python rktl_autonomy/train_sb3.py \
    --algorithm PPO \
    --timesteps 200000 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --eval-freq 20000

# Load and continue training
python rktl_autonomy/train_sb3.py \
    --load-model ./models/PPO_final_model \
    --timesteps 50000

# Render during evaluation
python rktl_autonomy/train_sb3.py --render
```

### Available Options

- `--algorithm`: PPO, DQN, A2C, SAC
- `--timesteps`: Total training timesteps
- `--learning-rate`: Learning rate (default: 3e-4)
- `--batch-size`: Batch size (default: 64)
- `--eval-freq`: Evaluation frequency (default: 10000)
- `--render`: Render during evaluation
- `--load-model`: Path to load existing model
- `--save-dir`: Directory to save models

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project directory
   cd /Users/hhatami/rocket_league_2
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **ROS Connection Issues**
   ```bash
   # Check ROS topics
   ros2 topic list
   ros2 topic echo /FieldState
   
   # Make sure simulator is running
   ros2 run rktl_simulator simulator.py
   ```

3. **CUDA/GPU Issues**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Memory Issues**
   ```python
   # Reduce batch size and episode length
   env = RocketLeagueSB3Env(max_episode_steps=500)
   model = PPO("MlpPolicy", env, batch_size=32)
   ```

### Performance Tips

1. **Faster Training**
   - Use vectorized environments
   - Reduce episode length for quick iterations
   - Use GPU if available

2. **Better Results**
   - Tune hyperparameters
   - Adjust reward function
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

requirements.txt                    # Python dependencies
setup.py                          # Package setup
README_SB3_INTEGRATION.md         # Detailed documentation
```

## Next Steps

1. **Start with Quick Examples**: Run `examples/quick_start_training.py`
2. **Try Different Algorithms**: Experiment with PPO, DQN, A2C, SAC
3. **Tune Hyperparameters**: Use the training script options
4. **ROS Integration**: Train with real simulation data
5. **Customize**: Modify reward functions and environments

## Support

- Check the detailed documentation in `README_SB3_INTEGRATION.md`
- View training progress with TensorBoard
- Experiment with different algorithms and parameters
- Integrate with your ROS 2 simulation system

Happy training! 🚀
