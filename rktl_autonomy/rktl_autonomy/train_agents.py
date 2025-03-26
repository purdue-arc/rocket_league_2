import gymnasium as gym
import training_env

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback

models_dir = r"~/models/"
logs_dir = r"~/logs/"
TIMESTEPS = 500_000
EVAL_FREQ = 10_000

env = gym.make("training_env/TouchBall-v0", render_mode="rgb_array")
eval_env = gym.make("training_env/TouchBall-v0")
eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                             log_path= logs_dir, eval_freq=EVAL_FREQ,
                             render=False)

model = DQN("MultiInputPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)

vec_env = model.get_env()
obs = vec_env.reset()
#model.save(f"{models_dir}/{TIMESTEPS}")
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()


vec_env.close()