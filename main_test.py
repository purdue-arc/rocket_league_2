from stable_baselines3 import PPO
from car_env import CarSoccerEnv

render_env = CarSoccerEnv(render_mode="human")
model = PPO.load("car_ai_model", device="cpu")

obs, _ = render_env.reset()
for _ in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = render_env.step(action)
    
    if terminated or truncated:
        obs, _ = render_env.reset()