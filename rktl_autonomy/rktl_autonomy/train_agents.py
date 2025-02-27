import gymnasium as gym
import training_env

from stable_baselines3 import PPO, A2C, DQN

models_dir = r"~/models"
TIMESTEPS = 500_000

env = gym.make("training_env/TouchBall-v0", render_mode="rgb_array")

model = DQN("MultiInputPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=TIMESTEPS)

vec_env = model.get_env()
obs = vec_env.reset()
model.save(f"{models_dir}/{TIMESTEPS}")
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()


vec_env.close()