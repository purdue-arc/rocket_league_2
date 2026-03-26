import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from car_env import CarSoccerEnv

# Renaming CustomSmallCNN to ImageFeatureExtractor to clarify its role
class ImageFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        # observation_space for 'image' is now (channels, h, w)
        self.n_input_channels = observation_space.shape[0]
        self.h, self.w = observation_space.shape[1], observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_input = th.zeros(1, self.n_input_channels, self.h, self.w)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # The input 'observations' here is already the 'image' tensor
        return self.linear(self.cnn(observations))

# Custom MultiInputFeatureExtractor that uses ImageFeatureExtractor for 'image' and a Linear layer for 'angle_error'
class CustomMultiInputFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        extractors = {}

        total_concat_size = 0
        # Process the 'image' part with ImageFeatureExtractor
        image_space = observation_space["image"]
        image_extractor = ImageFeatureExtractor(image_space) # Create an instance of our custom CNN
        extractors["image"] = image_extractor
        total_concat_size += image_extractor.features_dim

        # Process the 'angle_error' part with a simple linear layer
        angle_error_space = observation_space["angle_error"]
        angle_extractor = nn.Sequential(nn.Linear(angle_error_space.shape[0], 16), nn.ReLU()) # Small linear layer
        extractors["angle_error"] = angle_extractor
        total_concat_size += 16 # Output size of this linear layer

        self.extractors = nn.ModuleDict(extractors)

        # Final linear layer to combine features to the desired features_dim
        self.linear = nn.Sequential(nn.Linear(total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        # Assuming observations is a dictionary of tensors
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Concatenate all extracted features
        concatenated_features = th.cat(encoded_tensor_list, dim=1)
        return self.linear(concatenated_features)

if __name__ == "__main__":
    num_cpu = 8
    env = make_vec_env(CarSoccerEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=CustomMultiInputFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MultiInputPolicy", # Use MultiInputPolicy for Dict observation spaces
        env,
        policy_kwargs=policy_kwargs,
        device="auto",
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64
    )

    print("Starting training with Custom MultiInput CNN...")
    model.learn(total_timesteps=500_000)
    model.save("car_ai_spatial_model")