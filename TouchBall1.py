
import numpy as np

import gymnasium as gym
import math
from gymnasium import spaces
import pygame

class TouchBallNoPhysicsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}
    action_dist = {0: 0, 1: 0, 2: 0}

    def __init__(self, field_width=304.8, field_height=426.72, render_mode = None):
        self.field_width = field_width
        self.field_height = field_height
        self.turn_speed = 5
        self.move_speed = 5
        self.close_enough_radius = 15
        
        self.car_width = 10
        self.car_height = 20
        self.ball_radius = 10

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agentPos": spaces.Box(low=np.array([0, 0]), high=np.array([self.field_width, self.field_height]), dtype=float),
                "agentAngle" : spaces.Box(low=np.array([0]), high=np.array([360]), dtype=float),
                "ballPos": spaces.Box(low=np.array([0, 0]), high=np.array([self.field_width, self.field_height]), dtype=float),
            }
        )
        
        # self.observation_space = spaces.Tuple([
        #         spaces.Box(low=np.array([0, 0]), high=np.array([self.field_width, self.field_height]), dtype=float),
        #         spaces.Box(low=np.array([0]), high=np.array([360]), dtype=float),
        #         spaces.Box(low=np.array([0, 0]), high=np.array([self.field_width, self.field_height]), dtype=float),
        # ])

        self._agent_location = np.array([0, 0], dtype=float)
        self._agent_angle = np.array([0], dtype=float)
        self._ball_location = np.array([0, 0], dtype=float)

        self.action_space = spaces.Discrete(3, start=0)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agentPos": self._agent_location, "agentAngle": self._agent_angle, "ballPos": self._ball_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._ball_location, ord=1
            )
        }
        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([self.np_random.uniform(0, self.field_width), self.np_random.uniform(0, self.field_height)])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._ball_location = np.array([self.np_random.uniform(0, self.field_width), self.np_random.uniform(0, self.field_height)])

        # Random agent orientation
        self._agent_angle = np.array([self.np_random.uniform(0, 360)])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):

        prev_car_location = self._agent_location
        prev_car_angle = self._agent_angle
        self.action_dist[int(action)] += 1
        
        # Calculate angle to the ball
        ball_direction = np.array([self._ball_location[0] - self._agent_location[0], self._ball_location[1] - self._agent_location[1]])
        angle_to_ball = math.degrees(math.atan2(ball_direction[1], ball_direction[0]))
        angle_to_ball = angle_to_ball % 360  # Normalize to [0, 360)
        
        # Calculate the difference between the agent's current angle and the angle to the ball
        angle_diff_before = min(abs(self._agent_angle - angle_to_ball), abs(self._agent_angle - angle_to_ball + 360), abs(self._agent_angle - angle_to_ball - 360))
    
        # Perform action
        if (int(action) == Actions.TURN_L.value):
            self._agent_angle -= self.turn_speed            
    
        elif (int(action) == Actions.TURN_R.value):
            self._agent_angle += self.turn_speed
    
        elif (int(action) == Actions.FORWARD.value):
            delta_x = self.move_speed * math.cos(math.radians(self._agent_angle))
            delta_y = self.move_speed * math.sin(math.radians(self._agent_angle))
            self._agent_location += np.array([delta_x, delta_y])
            self._agent_location[0] = np.clip(self._agent_location[0], 0, self.field_width) 
            self._agent_location[1] = np.clip(self._agent_location[1], 0, self.field_height)
    
        # Normalize agent angle to [0, 360)
        self._agent_angle = self._agent_angle % 360
    
        # Calculate new difference after the action
        angle_diff_after = min(abs(self._agent_angle - angle_to_ball), abs(self._agent_angle - angle_to_ball + 360), abs(self._agent_angle - angle_to_ball - 360))
        
        # Check if Terminated
        prev_dist = np.linalg.norm(prev_car_location - self._ball_location)
        dist = np.linalg.norm(self._agent_location - self._ball_location)
        terminated = dist <= self.close_enough_radius
    
        # Rewards
        reward = 0
        if terminated:
            reward += 10000
        if dist > prev_dist:
            reward += 100
        # if dist > prev_dist:
        #     reward -= 10
        if angle_diff_after < angle_diff_before:  # Reward if the agent's angle is closer to the target angle
            reward += 10
        if angle_diff_after > angle_diff_before:
            reward -= 200 
    
        point = self._agent_location + np.array([20*math.cos(math.radians(self._agent_angle)), 20*math.sin(math.radians(self._agent_angle))])
        rectangle = np.array([0, 0, self.field_width, self.field_height])
        is_in_rectangle = (rectangle[0] <= point[0] <= rectangle[2]) and (rectangle[1] <= point[1] <= rectangle[3])
        # if (not is_in_rectangle):
        #     reward -= 500

    
        observation = self._get_obs()
        info = self._get_info()
    
        return observation, reward, terminated, False, info

    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.field_width, self.field_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
        canvas = pygame.Surface((self.field_width, self.field_height))
        canvas.fill((255, 255, 255))
        
        # Drawing the ball
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._ball_location.tolist(),
            self.ball_radius,
        )
        
        # Drawing the car
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._agent_location[0] - self.car_width/2,
                self._agent_location[1] - self.car_height/2,
                self.car_width,
                self.car_height
            ),
        )
    
        # Drawing the angle
        start = pygame.Vector2(self._agent_location[0], self._agent_location[1])
        end = pygame.Vector2(20*math.cos(math.radians(self._agent_angle)), 20*math.sin(math.radians(self._agent_angle)))
        self.draw_arrow(canvas, start, start + end, pygame.Color("dodgerblue"))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
    
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def draw_arrow(self, surface: pygame.Surface, start: pygame.Vector2, end: pygame.Vector2,
        color: pygame.Color, body_width: int = 10, head_width: int = 20, head_height: int = 12):
        """Draw an arrow between start and end with the arrow head at the end.
    
        Args:
            surface (pygame.Surface): The surface to draw on
            start (pygame.Vector2): Start position
            end (pygame.Vector2): End position
            color (pygame.Color): Color of the arrow
            body_width (int, optional): Defaults to 2.
            head_width (int, optional): Defaults to 4.
            head_height (float, optional): Defaults to 2.
        """
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height
    
        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start
    
        pygame.draw.polygon(surface, color, head_verts)
    
        # Stop weird shapes when the arrow is shorter than arrow head
        if arrow.length() >= head_height:
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start
    
            pygame.draw.polygon(surface, color, body_verts)

# # Test the render method
# if __name__ == "__main__":
#     env = TouchBallNoPhysicsEnv(render_mode="human")
    
#     running = True
#     clock = pygame.time.Clock()

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         env._render_frame()  # Call your _render_frame() method
#         clock.tick(24)  # Limit frame rate to 60 FPS

#     pygame.quit()