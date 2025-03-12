import gymnasium
import numpy as np
from gymnasium import spaces

from rktl_simulator.rktl_simulator.simulatorPoint import (FIELD_HEIGHT,
                                                          FIELD_WIDTH,
                                                          GOAL_DEPTH,
                                                          PointGame)


class CustomEnvironment(gymnasium.Env):
    def __init__(self):
        super(CustomEnvironment, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(8,), dtype=np.float64)
        # Define the action space (2 continuous outputs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.game = PointGame(carStartList=[
            [True, (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 2]
        ])
        def step(self, action):
            pass
            return observation, reward, done, info
        def reset(self):
            pass
            return observation
        def render(self, mode = 'human'):
            pass
        def close(self):
            pass