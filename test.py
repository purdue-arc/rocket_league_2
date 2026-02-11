import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "CartPole-v1"
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = DQN("MlpPolicy", vec_env, verbose=1, device="cuda")
    model.learn(total_timesteps=25_000)

    obs = vec_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render()