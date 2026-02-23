import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
from simulator import Game, FIELD_WIDTH, FIELD_HEIGHT, GOAL_DEPTH, SIDE_WALL, GOAL_HEIGHT, CAR_TURN, MAX_SPEED
import time

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

        # 10 dim: carX, carY, carAng, carXVel, carYVel, ballX, ballY, ballXVel, ballYVel, angle_error
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.rewardInfo = {'amtTouch': 0, 'prevDist': -1.0, 'prevAngleError': None, 'time': 0}

        self.physics_steps_per_gym_step = 1
        self.dt = 0.1 / self.physics_steps_per_gym_step

    def _compute_angle_error(self, throttle=None):
        """
        Compute the minimum angle the car must turn to face the ball,
        accounting for whether it's driving forward or backward.
        Returns angle error in radians, normalized to -pi to pi.
        """
        car = self.game.cars[0]
        ball = self.game.ball

        dx = ball.getPos().x - car.getPos().x
        dy = ball.getPos().y - car.getPos().y
        angle_to_ball = np.arctan2(dy, dx)  # radians, -pi to pi

        car_angle = np.radians(car.getAngle())

        # Forward angle error
        forward_error = angle_to_ball - car_angle
        forward_error = (forward_error + np.pi) % (2 * np.pi) - np.pi

        # Backward angle error (car rear facing ball)
        backward_error = angle_to_ball - (car_angle + np.pi)
        backward_error = (backward_error + np.pi) % (2 * np.pi) - np.pi

        if throttle is not None and abs(throttle) > 0.1:
            # Use whichever error matches the direction of travel
            angle_error = backward_error if throttle < 0 else forward_error
        else:
            # Pick whichever orientation is closer to the ball
            angle_error = forward_error if abs(forward_error) < abs(backward_error) else backward_error

        return angle_error

    def _get_obs(self, throttle=None):
        car = self.game.cars[0]
        ball = self.game.ball

        angle_error = self._compute_angle_error(throttle)

        return np.array([
            (car.getPos().x / FIELD_WIDTH) * 2 - 1,
            (car.getPos().y / FIELD_HEIGHT) * 2 - 1,
            (car.getAngle() / CAR_TURN) * 2 - 1,
            (car.getVelocity().x / MAX_SPEED) * 2 - 1,
            (car.getVelocity().y / MAX_SPEED) * 2 - 1,
            (ball.getPos().x / FIELD_WIDTH) * 2 - 1,
            (ball.getPos().y / FIELD_HEIGHT) * 2 - 1,
            (ball.getVelocity().x / MAX_SPEED) * 2 - 1,
            (ball.getVelocity().y / MAX_SPEED) * 2 - 1,
            angle_error / np.pi,  # normalized -1 to 1
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rewardInfo = {'amtTouch': 0, 'prevDist': -1, 'prevAngleError': None, 'time': 0}
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        throttle = float(action[0])
        steer = float(action[1])

        self.game.inputs[0] = [throttle, steer]

        for _ in range(self.physics_steps_per_gym_step):
            self.game.cars[0].update(self.game.inputs[0])
            self.game.gameSpace.step(self.dt)

        obs = self._get_obs(throttle=throttle)
        carX, carY, carAng, carXVel, carYVel, ballX, ballY, ballXVel, ballYVel, angle_error_norm = obs

        reward = 0
        terminated = False

        car_pos = self.game.cars[0].getPos()
        ball_pos = self.game.ball.getPos()
        dist_world = np.linalg.norm([car_pos.x - ball_pos.x, car_pos.y - ball_pos.y])

        angle_error = angle_error_norm * np.pi  # back to radians

        # --- Reward angle error improvement ---
        if self.rewardInfo['prevAngleError'] is not None:
            angle_delta = abs(self.rewardInfo['prevAngleError']) - abs(angle_error)
            if angle_delta > 0:
                reward += 10.0   # turning toward ball
            else:
                reward -= 5.0   # punish turning away (softer)
        
        reward -= abs(angle_error) / np.pi * 5.0 

        self.rewardInfo['prevAngleError'] = angle_error

        # --- Reward distance improvement ---
        if self.rewardInfo['prevDist'] >= 0:
            if self.rewardInfo['prevDist'] > dist_world:
                reward += 1.0
            else:
                reward -= 5.0

        self.rewardInfo['prevDist'] = dist_world
        self.rewardInfo['time'] += 1

        if(abs(throttle) < 0.05):
            reward -= 100.0

        # --- Touch reward ---
        if dist_world < 20:
            self.rewardInfo['amtTouch'] += 1
            reward += 1000
            terminated = True

        # --- Out of bounds ---
        if not(-0.9 < carX < 0.9) or not(-0.9 < carY < 0.9):
            reward -= 100
            terminated = True

        # --- Timeout ---
        if self.rewardInfo['time'] > 1_000:
            reward -= 100
            terminated = True

        reward -= 1

        if self.render_mode == "human":
            self.render()
            print('Distance:', dist_world)
            print('Angle error (deg):', np.degrees(angle_error))
            print('Current Reward:', reward)
            time.sleep(0.01)

        return obs, reward, terminated, False, {}

    def render(self):
        self.game.screen.fill(pygame.Color("white"))
        self.game.gameSpace.debug_draw(self.game.draw_options)

        car = self.game.cars[0]
        ball = self.game.ball
        start_pos = car.getPos()

        # Red: car facing direction
        angle_rad = np.radians(car.getAngle())
        facing_end = (
            start_pos.x + np.cos(angle_rad) * 50,
            start_pos.y + np.sin(angle_rad) * 50
        )
        pygame.draw.line(self.game.screen, (255, 0, 0), (start_pos.x, start_pos.y), facing_end, 3)

        # Green: direction to ball
        dx = ball.getPos().x - start_pos.x
        dy = ball.getPos().y - start_pos.y
        angle_to_ball = np.arctan2(dy, dx)
        target_end = (
            start_pos.x + np.cos(angle_to_ball) * 50,
            start_pos.y + np.sin(angle_to_ball) * 50
        )
        pygame.draw.line(self.game.screen, (0, 200, 0), (start_pos.x, start_pos.y), target_end, 2)

        # Blue: velocity direction
        vel = car.getVelocity()
        if vel.length > 0.1:
            vel_end = (
                start_pos.x + vel.x * 5,
                start_pos.y + vel.y * 5
            )
            pygame.draw.line(self.game.screen, (0, 0, 255), (start_pos.x, start_pos.y), vel_end, 3)

        pygame.display.set_caption(f"AI Training - Score L:{self.game.leftscore} R:{self.game.rightscore}")
        pygame.display.update()

    def close(self):
        pygame.quit()