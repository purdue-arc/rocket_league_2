import gymnasium as gym
import training_env

from stable_baselines3 import PPO, A2C, DQN

model_path = r"~\models\2000000.zip"

env = gym.make("gymnasium_env/TouchBall-v0", render_mode="rgb_array")

model = DQN.load(model_path, env=env)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10_000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()

vec_env.close()