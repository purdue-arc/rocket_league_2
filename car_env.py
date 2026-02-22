import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
from simulator import Game, FIELD_WIDTH, FIELD_HEIGHT, GOAL_DEPTH, SIDE_WALL, GOAL_HEIGHT, CAR_TURN, MAX_SPEED

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

        # 2 dim -1->1 float space, contains velocity, and steering angle
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # 9 dim -inf -> inf float space 
        # carX, carY, carAng, carXVel, carYVel, ballX, ballY, ballXVel, ballYVel
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.rewardInfo = {'amtTouch': 0}

        self.physics_steps_per_gym_step = 1
        self.dt = 0.1 / self.physics_steps_per_gym_step

    def _get_obs(self):
        car = self.game.cars[0]
        ball = self.game.ball
        return np.array([
            (car.getPos().x / FIELD_WIDTH) * 2 - 1,
            (car.getPos().y / FIELD_HEIGHT) * 2 - 1,
            (car.getAngle() / CAR_TURN) * 2 - 1,
            (car.getVelocity().x / MAX_SPEED) * 2 - 1,  # FIX: was ball velocity
            (car.getVelocity().y / MAX_SPEED) * 2 - 1,
            (ball.getPos().x / FIELD_WIDTH) * 2 - 1,
            (ball.getPos().y / FIELD_HEIGHT) * 2 - 1,
            (ball.getVelocity().x / MAX_SPEED) * 2 - 1,
            (ball.getVelocity().y / MAX_SPEED) * 2 - 1
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        # throttle = max(-1, min(1, action[0]))
        # steer = max(-1, min(1, action[1]))

        throttle = float(action[0])
        steer = float(action[1])
        

        # Ensure the game knows EXACTLY what the AI wants
        self.game.inputs[0] = [throttle, steer]

        for _ in range(self.physics_steps_per_gym_step):
            self.game.cars[0].update(self.game.inputs[0])
            self.game.gameSpace.step(self.dt)

        obs = self._get_obs()

        carX, carY, carAng, carXVel, carYVel, ballX, ballY, ballXVel, ballYVel = obs
        
        reward = 0
        terminated = False
        

        car_pos = self.game.cars[0].getPos()
        ball_pos = self.game.ball.getPos()
        dist_world = np.linalg.norm([car_pos.x - ball_pos.x, car_pos.y - ball_pos.y])

        reward -= dist_world / (FIELD_WIDTH * 100)  # normalize sensibly
        reward += self.rewardInfo['amtTouch'] * 100

        if dist_world < 30:  # world-space pixel threshold
            self.rewardInfo['amtTouch'] += 1

        if not(-0.9 < carX < 0.9) or not(-0.9 < carY < 0.9):
            reward -= 10
            terminated = True

        # elif ballX < GOAL_DEPTH:
        #     # reward -= 10
        #     terminated = True

        if self.render_mode == "human":
            self.render()
            print('Distace:', dist_world)
            print('Amount Touched:', self.rewardInfo['amtTouch'])
            print('Current Reward:', reward)
            print('CarX: ', carX)
            print('CarY: ', carY)
            

        return obs, reward, terminated, False, {}

    def render(self):
        self.game.screen.fill(pygame.Color("white"))
        self.game.gameSpace.debug_draw(self.game.draw_options)
        pygame.display.set_caption(f"AI Training - Score L:{self.game.leftscore} R:{self.game.rightscore}")
        pygame.display.update()

    def close(self):
        pygame.quit()