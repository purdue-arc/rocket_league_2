from simulator import Car, Ball, BALL_POS, CAR_POS, FIELD_HEIGHT, GOAL_DEPTH, FIELD_WIDTH

import pygame
import pymunk

import rclpy
from rktl_interfaces.msg import Field, Pose, CarAction

class PointGame():
    def __init__(
        self,
        carStartList:list[tuple[bool, float, float, float] | tuple[bool, float, float]] = CAR_POS,
        ballPosition:tuple[float, float] = BALL_POS
    ):
        """Constructor method

        :param carStartList: Positions of cars, defaults to CAR_POS
        :type carStartList: list[tuple[bool, float, float, float]  |  tuple[bool, float, float]], optional
        :param ballPosition: Position of the ball, defaults to BALL_POS
        :type ballPosition: tuple[float, float], optional
        """        """Constructor method"""
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH + GOAL_DEPTH, FIELD_HEIGHT))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()

        self.leftscore = 0
        self.rightscore = 0
        self.ticks = 60
        self.ballPosition = ballPosition
        self.carStartList = carStartList
        self.cars = []
        self.inputs = []
        self.exit = False

        self.gameSpace = pymunk.Space()
        
    def run(self, steps:int=10):
        """Main logic function to keep track of gamestate. Steps 0.1 seconds with each SPACE key press.
        :param steps: Number of steps taken per 0.1 seconds when SPACE key is pressed
        :type steps: int
        """
        self.ball = Ball(self.ballPosition[0], self.ballPosition[1], self.gameSpace, pymunk.vec2d.Vec2d(10,0))
        self.cars.append(Car(2 * (FIELD_WIDTH + GOAL_DEPTH) / 3, FIELD_HEIGHT / 2, self.gameSpace, 180))

        self.screen.fill(pygame.Color("white"))
        self.gameSpace.debug_draw(self.draw_options)
        pygame.display.update()
        while not self.exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
            # User input
            self.pressed = pygame.key.get_pressed()

            if self.pressed[pygame.K_SPACE]:
                for _ in range(steps):
                    self.cars[0].update([1,0])
                    self.screen.fill(pygame.Color("white"))
                    self.gameSpace.debug_draw(self.draw_options)
                    pygame.display.update()
                    self.gameSpace.step(0.1/steps)
            pygame.time.wait(100)