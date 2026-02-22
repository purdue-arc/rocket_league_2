from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from car_env import CarSoccerEnv

if __name__ == "__main__":
    num_cpu = 8
    
    env = make_vec_env(CarSoccerEnv, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy",
        env, 
        device="cpu",
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,
        learning_rate=1e-4,
        verbose=1
    )

    model.learn(total_timesteps=500_000)
    model.save("car_ai_model")