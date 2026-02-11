from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from car_env import CarSoccerEnv

if __name__ == "__main__":
    num_cpu = 48
    
    env = make_vec_env(CarSoccerEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy", 
        env, 
        device="cuda",
        n_steps=2048*4,
        batch_size=512,
        learning_rate=3e-4,
        verbose=1
    )

    model.learn(total_timesteps=1_000_000)
    model.save("car_ai_model")