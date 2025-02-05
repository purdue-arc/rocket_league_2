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
        
        self.node = rclpy.create_node("simNode")
        self.publisher = self.node.create_publisher(Field, "simTopic", 10)
        self.subscriber = self.node.create_subscription(CarAction, "aiTopic", self.runAStep, 10)
    
    def runAStep(self, msg: CarAction):
        """The callback function for the subscriber

        :param msg: The message sent in
        :type msg: CarAction
        """        
        steps = 10
        for _ in range(steps):
            self.cars[msg.id].update([msg.throttle, msg.steer])
            pygame.display.update()
            self.gameSpace.step(0.1/steps)
        msgOut = Field()
        msgOut.ball_pose.id = -1
        (msgOut.ball_pose.x, msgOut.ball_pose.y) = self.ball.getPos()
        msg.ball_pose.angle = 0.0
        for i, c in enumerate(self.cars):
            tempPose = Pose()
            tempPose.id = i
            (tempPose.x, tempPose.y) = c.getPos()
            tempPose.angle = c.getAngle()
            if c.team:
                msgOut.team1_poses.append(tempPose)
            else:
                msgOut.team2_poses.append(tempPose)
        self.publisher.publish(msgOut)

class MessagePublisher(rclpy.Node):
    def __init__(
        self,
        publisherName: str,
        publisherTopic: str,
        gameInstance: PointGame,
        queueSize: int = 10,
        timed: bool = False,
        timerPeriod: int = 10
    ):
        super().__init__(publisherName)
        if timed:
            self.timer = self.create_timer(timerPeriod, self.timer_callback)
        self.publisher_ = self.create_publisher(Field, publisherTopic, queueSize)
        self.gameInstance = gameInstance
    
    def timer_callback(self):
        msg = Field()
        msg.ball_pose.id = -1
        msg.ball_pose.x = self.gameInstance.ball.getPos().x
        msg.ball_pose.y = self.gameInstance.ball.getPos().y
        msg.ball_pose.angle = 0.0
        for i, c in enumerate(self.gameInstance.cars):
            tempPose = Pose()
            tempPose.id = i
            (tempPose.x, tempPose.y) = c.getPos()
            tempPose.angle = c.getAngle()
            if c.team:
                msg.team1_poses.append(tempPose)
            else:
                msg.team2_poses.appent(tempPose)
        self.publisher_.publish(msg)
