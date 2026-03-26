from stable_baselines3 import PPO
import pygame
from car_env import CarSoccerEnv

model = PPO.load("car_ai_spatial_model")

# Create the environment with rendering enabled
# The render_mode="human" will open a Pygame window to visualize the simulation.
env = CarSoccerEnv(render_mode="human")

# Reset the environment
obs, info = env.reset()

# Run the agent for a few episodes or steps
print("Starting evaluation of the trained AI...")
for _ in range(100):  # Run for 5 episodes
    done = False
    total_reward = 0
    obs, info = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        # Optional: Add a small delay for better visualization
        pygame.time.wait(15)
    print(f"Episode finished with total reward: {total_reward}")

env.close()
print("Evaluation complete.")
