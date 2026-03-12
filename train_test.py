import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from car_env import CarSoccerEnv

class CustomSmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # Should be 5
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    num_cpu = 8
    env = make_vec_env(CarSoccerEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=CustomSmallCNN, # Use the small grid extractor
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False, # Required for float32 observations
        net_arch=dict(pi=[128, 128], vf=[128, 128]) # 'vf' instead of 'qf' for PPO
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="mps", 
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64
    )

    print("Starting training with 20x20 Grid CNN...")
    model.learn(total_timesteps=500_000)
    model.save("car_ai_spatial_model")