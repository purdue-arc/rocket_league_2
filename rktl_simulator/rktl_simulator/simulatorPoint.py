import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rktl_simulator.simulator import BALL_POS, CAR_POS, Game

class PointGame(gym.Env):
    def __init__(
        self,
        team: int,
        carStartList = CAR_POS,
        ballPosition = BALL_POS,
    ):
        super().__init__()

        assert team in (0, 1)
        self.team = team

        self.startCarList = carStartList.copy()
        self.ballStart = (ballPosition[0], ballPosition[1])

        self.action_space = spaces.Box(
            low=np.array([-1, -30, -1, -30], dtype=np.float32),
            high=np.array([1, 30, 1, 30], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(33,),
            dtype=np.float32,
        )

        self.game = Game(carStartList, ballPosition)

    def reset(self, *, seed=None, options=None):
        self.game = Game(self.startCarList, self.ballStart)
        return self.create_observation_space(), {}

    def close(self):
        pass

    def create_observation_space(self):
        scores = [self.game.leftscore, self.game.rightscore]
        if self.team == 1:
            scores = [self.game.rightscore, self.game.leftscore]

        time_left = self.game.time_left

        ball = self.game.ball
        ball_pos = ball.getPos()
        ball_velocity = ball.getVelocity()

        ball_pos_list = [ball_pos.x, ball_pos.y, ball_pos.angle]
        ball_velocity_list = [ball_velocity.x, ball_velocity.y, ball_velocity.angle]

        obs_space = scores + [time_left] + ball_pos_list + ball_velocity_list

        for i in range(4):
            car = self.game.cars[i]

            car_pos = car.body.position
            car_velocity = car.body.velocity

            car_pos_list = [car_pos[0], car_pos[1], car.body.angle]
            car_velocity_list = [car_velocity.x, car_velocity.y, car_velocity.angle]

            obs_space += car_pos_list + car_velocity_list

        return np.array(obs_space, dtype=np.float32)

    def calculateReward(self):
        return random.random() * 5 - 5

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        self.game.updateCars(self.team, [action[0], action[1]], [action[2], action[3]])
        self.game.step()

        obs = self.create_observation_space()
        reward = float(self.calculateReward())
        done = self.game.time_left <= 0
        return obs, reward, done, False, {}

def main():
    game = PointGame(team=0)
    obs, _ = game.reset()
    print("Initial observation:", obs)

if __name__ == "__main__":
    main()