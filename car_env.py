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

        self.grid_size = (20, 20)  # Resolution of your boxes
        
        # 5 Channels: 
        # 0: Car Pos, 1: Ball Pos, 2: Car Velocity (Norm), 3: Car Angle, 4: Ball Velocity
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(5, self.grid_size[0], self.grid_size[1]), 
            dtype=np.float32
        )

        self.rewardInfo = {'amtTouch': 0, 'prevDist': -1.0, 'prevAngleError': None, 'time': 0}

        self.physics_steps_per_gym_step = 1
        self.dt = 0.1 / self.physics_steps_per_gym_step

    def _get_obs(self, throttle=None):
        grid = np.zeros((5, self.grid_size[0], self.grid_size[1]), dtype=np.float32)
        
        car = self.game.cars[0]
        ball = self.game.ball

        # Map world coordinates to grid indices
        def to_grid(pos_x, pos_y):
            # Normalizing based on FIELD_WIDTH and HEIGHT from simulator.py
            col = int((pos_x / (FIELD_WIDTH + GOAL_DEPTH)) * (self.grid_size[1] - 1))
            row = int((pos_y / FIELD_HEIGHT) * (self.grid_size[0] - 1))
            return np.clip(row, 0, self.grid_size[0]-1), np.clip(col, 0, self.grid_size[1]-1)

        c_r, c_c = to_grid(car.getPos().x, car.getPos().y)
        b_r, b_c = to_grid(ball.getPos().x, ball.getPos().y)

        # Layer 0 & 1: Discrete Positions
        grid[0, c_r, c_c] = 1.0
        grid[1, b_r, b_c] = 1.0
        
        # Layer 2: Car Velocity (Scalar intensity at the car's location)
        grid[2, c_r, c_c] = car.getVelocity().length / MAX_SPEED
        
        # Layer 3: Car Angle (Normalized -1 to 1 at the car's location)
        grid[3, c_r, c_c] = np.radians(car.getAngle()) / np.pi
        
        # Layer 4: Ball Velocity
        grid[4, b_r, b_c] = ball.getVelocity().length / MAX_SPEED

        return grid
    
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rewardInfo = {'amtTouch': 0, 'prevDist': -1, 'prevAngleError': None, 'time': 0}
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        throttle = float(action[0])
        steer = float(action[1])

        self.game.inputs[0] = [throttle, steer]

        # 1. Step the physics simulation
        for _ in range(self.physics_steps_per_gym_step):
            self.game.cars[0].update(self.game.inputs[0])
            self.game.gameSpace.step(self.dt)

        # 2. Get the new spatial observation (The 3D "Boxes" grid)
        obs = self._get_obs(throttle=throttle)

        # 3. Calculate metrics directly from the simulation for rewards
        car = self.game.cars[0]
        ball = self.game.ball
        car_pos = car.getPos()
        ball_pos = ball.getPos()
        
        dist_world = np.linalg.norm([car_pos.x - ball_pos.x, car_pos.y - ball_pos.y])
        angle_error = self._compute_angle_error(throttle)

        reward = 0
        terminated = False

        # --- GOAL: Drive toward the ball ---
        
        # Reward 1: Distance Improvement (Progress)
        if self.rewardInfo['prevDist'] >= 0:
            # Positive reward for getting closer, negative for moving away
            dist_delta = self.rewardInfo['prevDist'] - dist_world
            reward += dist_delta * 2.0  # Weighted to encourage movement
        
        self.rewardInfo['prevDist'] = dist_world

        # Reward 2: Directional Alignment
        # Reward the agent for facing the ball, especially when throttle is applied
        alignment = np.cos(angle_error)
        if throttle > 0:
            reward += alignment * throttle * 5.0 # High reward for driving toward the ball

        # Reward 3: Speed Incentive
        # Punish idling to prevent the car from sitting still
        if abs(throttle) < 0.1:
            reward -= 5.0

        # --- Termination Logic ---

        # Success: Touching the ball
        if dist_world < 25: 
            reward += 500.0
            terminated = True

        # Failure: Out of Bounds
        # (Using normalized coordinates for boundary check)
        norm_x = (car_pos.x / (FIELD_WIDTH + GOAL_DEPTH)) * 2 - 1
        norm_y = (car_pos.y / FIELD_HEIGHT) * 2 - 1
        if not (-0.95 < norm_x < 0.95) or not (-0.95 < norm_y < 0.95):
            reward -= 100.0
            terminated = True

        # Failure: Timeout
        self.rewardInfo['time'] += 1
        reward -= 1
        if self.rewardInfo['time'] > 500:
            terminated = True
            reward -= 100

        # Constant pressure to finish quickly
        reward -= 0.5

        if self.render_mode == "human":
            self.render()
            print(f'Dist: {dist_world:.1f} | Angle Err: {np.degrees(angle_error):.1f} | Reward: {reward:.2f}')
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