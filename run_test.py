from stable_baselines3 import PPO
from car_env import CarSoccerEnv

env = CarSoccerEnv(render_mode="human")
model = PPO.load("car_ai_model")

obs, _ = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()