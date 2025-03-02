import gymnasium as gym
from rktl_autonomy.envs.sb3_no_physics import TouchBallNoPhysicsEnv

from stable_baselines3 import PPO, A2C

def main():

  TIMESTEPS = 200000

  env = TouchBallNoPhysicsEnv()

  model = A2C("MultiInputPolicy", env, verbose=1)
  model.learn(total_timesteps=TIMESTEPS)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      vec_env.render("human")
      # VecEnv resets automatically
      if done:
        obs = vec_env.reset()

  model.save(f"models/thebest")
  vec_env.close()