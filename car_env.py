import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
from simulator import Game, FIELD_WIDTH, FIELD_HEIGHT, GOAL_DEPTH, SIDE_WALL, GOAL_HEIGHT

class CarSoccerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode == "human":
            render = True
            print('rendering')
        else:
            render = False
        self.game = Game(render=render)
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.physics_steps_per_gym_step = 10
        self.dt = 0.1 / self.physics_steps_per_gym_step

    def _get_obs(self):
        car = self.game.cars[0]
        ball = self.game.ball
        return np.array([
            car.getPos().x, car.getPos().y, car.getAngle(),
            car.getVelocity().x, car.getVelocity().y,
            ball.getPos().x, ball.getPos().y,
            ball.getVelocity().x, ball.getVelocity().y
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        self.game.inputs[0] = [float(action[0]), float(action[1])]

        for _ in range(self.physics_steps_per_gym_step):
            self.game.updateObjects(walls=True, useKeys=False)
            self.game.gameSpace.step(self.dt)

        obs = self._get_obs()
        ball_x = obs[5]
        
        reward = 0
        terminated = False
        

        dist = np.linalg.norm(obs[0:2] - obs[5:7])
        reward -= dist * 0.01

        if ball_x > FIELD_WIDTH:
            reward += 100
            terminated = True
        elif ball_x < GOAL_DEPTH:
            reward -= 10
            terminated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def render(self):
        self.game.screen.fill(pygame.Color("white"))
        self.game.gameSpace.debug_draw(self.game.draw_options)
        pygame.display.set_caption(f"AI Training - Score L:{self.game.leftscore} R:{self.game.rightscore}")
        pygame.display.update()

    def close(self):
        pygame.quit()