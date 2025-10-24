"""
Rocket League Environment for Stable Baselines 3 Integration
This environment is optimized for SB3 training with proper observation/action spaces
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, Tuple, Any, Optional
from enum import Enum

class Actions(Enum):
    """Available actions for the agent"""
    TURN_R = 0
    TURN_L = 1
    FORWARD = 2

class RocketLeagueSB3Env(gym.Env):
    """
    Rocket League environment optimized for Stable Baselines 3 training.
    
    This environment provides:
    - Normalized observation space for better training
    - Discrete action space (3 actions)
    - Proper reward shaping for RL training
    - Episode termination conditions
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}
    
    def __init__(
        self, 
        field_width: float = 304.8, 
        field_height: float = 426.72, 
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        reward_scale: float = 1.0
    ):
        """
        Initialize the Rocket League environment for SB3 training.
        
        Args:
            field_width: Width of the field
            field_height: Height of the field  
            render_mode: Rendering mode ("human", "rgb_array", or None)
            max_episode_steps: Maximum steps per episode
            reward_scale: Scaling factor for rewards
        """
        super().__init__()
        
        # Environment parameters
        self.field_width = field_width
        self.field_height = field_height
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        
        # Movement parameters
        self.turn_speed = 5.0
        self.move_speed = 5.0
        self.close_enough_radius = 15.0
        
        # Visual parameters
        self.car_width = 10
        self.car_height = 20
        self.ball_radius = 10
        
        # State variables
        self._agent_location = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_angle = 0.0
        self._ball_location = np.array([0.0, 0.0], dtype=np.float32)
        self._episode_steps = 0
        
        # Action distribution tracking
        self.action_dist = {0: 0, 1: 0, 2: 0}
        
        # Define observation space (normalized)
        # Agent position, angle, ball position, distance to ball, angle to ball
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(3)
        
        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
    def _get_obs(self) -> np.ndarray:
        """
        Get normalized observation vector.
        
        Returns:
            Normalized observation: [agent_x, agent_y, agent_angle, ball_x, ball_y, distance, angle_to_ball]
        """
        # Normalize positions to [-1, 1]
        norm_agent_pos = np.array([
            (self._agent_location[0] / self.field_width) * 2 - 1,
            (self._agent_location[1] / self.field_height) * 2 - 1
        ], dtype=np.float32)
        
        norm_ball_pos = np.array([
            (self._ball_location[0] / self.field_width) * 2 - 1,
            (self._ball_location[1] / self.field_height) * 2 - 1
        ], dtype=np.float32)
        
        # Normalize angle to [-1, 1]
        norm_angle = (self._agent_angle / 180.0) - 1.0
        
        # Calculate distance and angle to ball
        distance = np.linalg.norm(self._agent_location - self._ball_location)
        norm_distance = min(distance / (np.sqrt(self.field_width**2 + self.field_height**2)), 1.0)
        
        # Calculate angle to ball
        ball_direction = self._ball_location - self._agent_location
        angle_to_ball = math.degrees(math.atan2(ball_direction[1], ball_direction[0]))
        angle_diff = self._agent_angle - angle_to_ball
        # Normalize angle difference to [-1, 1]
        norm_angle_diff = (angle_diff / 180.0) % 2.0 - 1.0
        
        return np.concatenate([
            norm_agent_pos,           # [agent_x, agent_y]
            [norm_angle],             # [agent_angle]
            norm_ball_pos,            # [ball_x, ball_y]
            [norm_distance],          # [distance_to_ball]
            [norm_angle_diff]         # [angle_to_ball]
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        distance = np.linalg.norm(self._agent_location - self._ball_location)
        return {
            "distance_to_ball": distance,
            "agent_position": self._agent_location.copy(),
            "ball_position": self._ball_location.copy(),
            "agent_angle": self._agent_angle,
            "episode_steps": self._episode_steps,
            "action_distribution": self.action_dist.copy()
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset episode counter
        self._episode_steps = 0
        
        # Reset action distribution
        self.action_dist = {0: 0, 1: 0, 2: 0}
        
        # Randomly place agent
        self._agent_location = np.array([
            self.np_random.uniform(0, self.field_width),
            self.np_random.uniform(0, self.field_height)
        ], dtype=np.float32)
        
        # Randomly place ball (ensure it's not too close to agent)
        min_distance = 50.0
        while True:
            ball_pos = np.array([
                self.np_random.uniform(0, self.field_width),
                self.np_random.uniform(0, self.field_height)
            ], dtype=np.float32)
            if np.linalg.norm(self._agent_location - ball_pos) > min_distance:
                self._ball_location = ball_pos
                break
        
        # Random agent orientation
        self._agent_angle = self.np_random.uniform(0, 360)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self._episode_steps += 1
        
        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
            
        self.action_dist[action] += 1
        
        # Store previous state for reward calculation
        prev_location = self._agent_location.copy()
        prev_angle = self._agent_angle
        self._prev_location = prev_location  # Store for reward calculations
        
        # Calculate angle to ball before action
        ball_direction = self._ball_location - self._agent_location
        angle_to_ball = math.degrees(math.atan2(ball_direction[1], ball_direction[0]))
        angle_diff_before = abs(self._agent_angle - angle_to_ball)
        if angle_diff_before > 180:
            angle_diff_before = 360 - angle_diff_before
        
        # Execute action
        if action == Actions.TURN_L.value:
            self._agent_angle -= self.turn_speed
        elif action == Actions.TURN_R.value:
            self._agent_angle += self.turn_speed
        elif action == Actions.FORWARD.value:
            # Move forward in current direction
            delta_x = self.move_speed * math.cos(math.radians(self._agent_angle))
            delta_y = self.move_speed * math.sin(math.radians(self._agent_angle))
            self._agent_location += np.array([delta_x, delta_y], dtype=np.float32)
            
            # Keep agent within field bounds
            self._agent_location[0] = np.clip(self._agent_location[0], 0, self.field_width)
            self._agent_location[1] = np.clip(self._agent_location[1], 0, self.field_height)
        
        # Normalize angle
        self._agent_angle = self._agent_angle % 360
        
        # Calculate new angle difference
        angle_diff_after = abs(self._agent_angle - angle_to_ball)
        if angle_diff_after > 180:
            angle_diff_after = 360 - angle_diff_after
        
        # Calculate distances
        prev_distance = np.linalg.norm(prev_location - self._ball_location)
        current_distance = np.linalg.norm(self._agent_location - self._ball_location)
        
        # Check termination conditions
        terminated = current_distance <= self.close_enough_radius
        truncated = self._episode_steps >= self.max_episode_steps
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_distance, current_distance, 
            angle_diff_before, angle_diff_after, 
            terminated
        )
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(
        self, 
        prev_distance: float, 
        current_distance: float,
        angle_diff_before: float,
        angle_diff_after: float,
        terminated: bool
    ) -> float:
        """
        Calculate comprehensive reward based on advanced Rocket League strategies.
        
        Args:
            prev_distance: Previous distance to ball
            current_distance: Current distance to ball
            angle_diff_before: Angle difference before action
            angle_diff_after: Angle difference after action
            terminated: Whether episode terminated
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # === OFFENSIVE PLAY REWARDS ===
        reward += self._calculate_offensive_rewards(prev_distance, current_distance, 
                                                  angle_diff_before, angle_diff_after, terminated)
        
        # === DEFENSIVE PLAY REWARDS ===
        reward += self._calculate_defensive_rewards(prev_distance, current_distance, terminated)
        
        # === TEAM COORDINATION REWARDS ===
        reward += self._calculate_team_coordination_rewards(prev_distance, current_distance)
        
        # === POSITIONAL AWARENESS REWARDS ===
        reward += self._calculate_positional_rewards(prev_distance, current_distance)
        
        # === BASIC EFFICIENCY ===
        reward -= 0.01  # Minimal step penalty (further reduced)
        
        return reward * self.reward_scale
    
    def _calculate_offensive_rewards(self, prev_distance: float, current_distance: float,
                                   angle_diff_before: float, angle_diff_after: float, 
                                   terminated: bool) -> float:
        """Calculate offensive play rewards."""
        reward = 0.0
        
        # a. Ball Control and Attacking
        if terminated:
            # Successfully reached the ball - base reward
            reward += 1000.0  # Increased from 100.0
            
            # Bonus for good approach angle (within 30 degrees of optimal)
            if angle_diff_after < 30:
                reward += 500.0  # High-quality approach (increased from 50.0)
            elif angle_diff_after < 60:
                reward += 250.0  # Decent approach (increased from 25.0)
            else:
                reward += 100.0  # Poor approach (increased from 10.0)
        
        # Reward for approaching ball intelligently
        distance_improvement = prev_distance - current_distance
        if distance_improvement > 0:
            # Bonus for good approach speed (not too fast, not too slow)
            current_speed = np.linalg.norm(self._agent_location - self._prev_location) if hasattr(self, '_prev_location') else 0
            if 3.0 <= current_speed <= 8.0:  # Optimal speed range
                reward += distance_improvement * 20.0  # High reward for good approach (increased from 5.0)
            else:
                reward += distance_improvement * 10.0  # Standard reward (increased from 2.0)
        
        # Reward for maintaining good angle to ball
        if angle_diff_after < 45:  # Good angle to ball
            reward += 10.0  # Increased from 2.0
        elif angle_diff_after < 90:  # Decent angle
            reward += 5.0  # Increased from 1.0
        
        # Penalty for poor positioning (too close without good angle)
        if current_distance < 20 and angle_diff_after > 90:
            reward -= 5.0  # Poor positioning penalty
        
        # Progressive rewards for getting closer to ball
        if current_distance < 50:
            reward += 20.0  # Very close to ball
        elif current_distance < 100:
            reward += 10.0  # Close to ball
        elif current_distance < 200:
            reward += 5.0   # Getting close
        
        return reward
    
    def _calculate_defensive_rewards(self, prev_distance: float, current_distance: float, 
                                   terminated: bool) -> float:
        """Calculate defensive play rewards."""
        reward = 0.0
        
        # Calculate field positions
        field_center_x = self.field_width / 2
        field_center_y = self.field_height / 2
        goal_x = 0  # Left goal
        goal_y = field_center_y
        
        # Distance from goal
        goal_distance = np.linalg.norm(self._agent_location - np.array([goal_x, goal_y]))
        ball_goal_distance = np.linalg.norm(self._ball_location - np.array([goal_x, goal_y]))
        
        # a. Goal Defense (Blocking Shots)
        if ball_goal_distance < 100:  # Ball is near our goal
            if goal_distance < 50:  # We're in defensive position
                reward += 50.0  # Good defensive positioning (increased from 10.0)
                
                # Bonus for being between ball and goal
                ball_to_goal = np.array([goal_x, goal_y]) - self._ball_location
                agent_to_goal = np.array([goal_x, goal_y]) - self._agent_location
                if np.dot(ball_to_goal, agent_to_goal) > 0:  # We're between ball and goal
                    reward += 75.0  # Excellent defensive positioning (increased from 15.0)
        
        # b. Clearance and Positioning for Defense
        if current_distance < 30:  # Close to ball
            # Check if we're in defensive half
            if self._agent_location[0] < field_center_x:  # In defensive half
                # Reward for clearing ball to offensive half
                ball_clearance = self._ball_location[0] - field_center_x
                if ball_clearance > 0:  # Ball is in offensive half
                    reward += 20.0  # Successful clearance
                else:
                    reward += 5.0  # Partial clearance
        
        # c. Back Positioning (Returning to Net)
        if goal_distance > 100:  # We're far from goal
            if ball_goal_distance < 150:  # Ball is near our goal
                # Penalty for being out of position when ball is near goal
                reward -= 10.0
            else:
                # Reward for returning to defensive position
                if goal_distance < 80:  # Good defensive position
                    reward += 5.0
        
        return reward
    
    def _calculate_team_coordination_rewards(self, prev_distance: float, current_distance: float) -> float:
        """Calculate team coordination rewards."""
        reward = 0.0
        
        # For now, we'll implement basic team awareness
        # In a full implementation, you'd have teammate positions
        
        # a. Teammate Proximity and Support
        field_center_x = self.field_width / 2
        field_center_y = self.field_height / 2
        
        # Reward for maintaining good field position
        center_distance = np.linalg.norm(self._agent_location - np.array([field_center_x, field_center_y]))
        if 50 < center_distance < 150:  # Good midfield position
            reward += 3.0
        
        # b. Avoiding Ball Chasing When Teammates Are Present
        # This would require teammate positions in a full implementation
        # For now, we'll penalize excessive ball chasing
        if current_distance < 20:  # Very close to ball
            # Check if we're moving too aggressively
            if hasattr(self, '_prev_location'):
                movement = np.linalg.norm(self._agent_location - self._prev_location)
                if movement > 10:  # Moving very fast toward ball
                    reward -= 2.0  # Penalty for over-chasing
        
        # c. Rotation and Positioning
        # Reward for maintaining good spacing
        if 30 < current_distance < 100:  # Good distance from ball
            reward += 2.0
        
        return reward
    
    def _calculate_positional_rewards(self, prev_distance: float, current_distance: float) -> float:
        """Calculate positional awareness rewards."""
        reward = 0.0
        
        field_center_x = self.field_width / 2
        field_center_y = self.field_height / 2
        
        # a. Midfield Play
        center_distance = np.linalg.norm(self._agent_location - np.array([field_center_x, field_center_y]))
        if 50 < center_distance < 120:  # Good midfield position
            reward += 2.0
        elif center_distance > 200:  # Too far from center
            reward -= 3.0
        
        # b. Time Management (Clock Awareness)
        # Reward for efficiency based on episode progress
        episode_progress = self._episode_steps / self.max_episode_steps
        if episode_progress > 0.8:  # Late in episode
            if current_distance < 50:  # Close to ball
                reward += 5.0  # Bonus for urgency
            else:
                reward -= 2.0  # Penalty for being far from action
        
        # c. Field Coverage
        # Reward for being in useful positions
        if (50 < self._agent_location[0] < self.field_width - 50 and 
            50 < self._agent_location[1] < self.field_height - 50):
            reward += 1.0  # Good field coverage
        
        return reward
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        """Render a single frame."""
        import pygame
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (int(self.field_width), int(self.field_height))
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((int(self.field_width), int(self.field_height)))
        canvas.fill((255, 255, 255))
        
        # Draw ball
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int(self._ball_location[0]), int(self._ball_location[1])),
            self.ball_radius
        )
        
        # Draw agent
        agent_rect = pygame.Rect(
            int(self._agent_location[0] - self.car_width/2),
            int(self._agent_location[1] - self.car_height/2),
            self.car_width,
            self.car_height
        )
        pygame.draw.rect(canvas, (255, 0, 0), agent_rect)
        
        # Draw direction arrow
        arrow_length = 20
        end_x = self._agent_location[0] + arrow_length * math.cos(math.radians(self._agent_angle))
        end_y = self._agent_location[1] + arrow_length * math.sin(math.radians(self._agent_angle))
        pygame.draw.line(
            canvas, 
            (0, 255, 0), 
            (int(self._agent_location[0]), int(self._agent_location[1])),
            (int(end_x), int(end_y)), 
            3
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
