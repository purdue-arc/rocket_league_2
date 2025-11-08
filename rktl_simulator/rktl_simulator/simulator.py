from math import cos, radians, sin, degrees

import pygame
import pymunk
import pymunk.pygame_util

#need to implement throttle and



# Field Specs
FIELD_WIDTH = 426.72
FIELD_HEIGHT = 304.8
GOAL_HEIGHT = 81.28
GOAL_DEPTH = 25.4
SIDE_WALL = (FIELD_HEIGHT - GOAL_HEIGHT) / 2
FIELD_FRICTION = 0.3
FIELD_ELASTICITY = 0.5
FIELD_COLOR = pygame.Color("white")

# Car Specs
CAR_SIZE = (16.5, 8.5)  # (Length, Width)
CAR_MASS = 5
CAR_SPEED = 5  # Impulse applied for forward/backward movement
CAR_TURN = 30  # Angular velocity for car turning
BRAKE_SPEED = 5  # Multiplier of CAR_SPEED for braking force
FREE_DECELERATION = 0.5  # Rate velocity decreases with no input
CAR_FRICTION = 0.5
CAR_COLOR = pygame.Color("green")
CAR_COLOR_ALT = pygame.Color("red")
CAR_POS = [[True, (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 3], [True, (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT/1.5], [False, 2 * (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 3,180], [False, 2 * (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 1.5,180]]


# Ball Specs
BALL_MASS = 0.1
BALL_RADIUS = 6.85 / 2
BALL_POS = (FIELD_WIDTH + GOAL_DEPTH) / 2, FIELD_HEIGHT / 2  # Starting position of the ball (x, y)
BALL_ELASTICITY = 1
BALL_FRICTION = 0.5
BALL_DECELERATION = 0.1
BALL_COLOR = pygame.Color("blue")

# Sim Limits
MAX_SPEED = 200  # Max speed limit of the cars


class Car:
    """Class used to define each car in the simulator.
    :param team: Boolean that determines which team the car is on (true=team1, false=team2)
    :type team: bool
    :param x: x coordinate for the starting position of the car
    :type x: float
    :param y: y coordinate for the starting position of the car
    :type y: float
    :param space: Contains the :class: `pymunk.space` class object for calculating physical interactions
    :type space: class: `pymunk.space`
    :param angle: Starting rotational angle (in degrees) for the car, where 0 degrees is East
    :type angle: float
    """

    def __init__(
            self,
            team: bool,
            x: float,
            y: float,
            space: pymunk.Space,
            angle: float,
            controllerKeys:tuple[int, int, int, int] = (
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_LEFT,
            pygame.K_RIGHT
        )
    ):
        """Constructor method"""
        self.controllerKeys = controllerKeys
        self.body = pymunk.Body(CAR_MASS, pymunk.moment_for_box(CAR_MASS, CAR_SIZE))
        self.body.position = (x, y)
        self.body.angle = radians(angle)

        self.shape = pymunk.Poly.create_box(self.body, CAR_SIZE)
        self.team = team
        if (self.team):
            self.shape.color = CAR_COLOR
        else:
            self.shape.color = CAR_COLOR_ALT
        self.shape.friction = CAR_FRICTION

        self.space = space
        self.space.add(self.body, self.shape)

        self.steering = 0.0  # Rate of steering
        self.reverse = 1  # Stores which direction car moves, 1 for forwards and -1 for backwards

    def keyUpdate(self, keys:pygame.key.ScancodeWrapper):
        """Calculates the needed motion of the car class called. Takes user keyboard inputs for controls
        :param keys: Passes :class: 'pygame.key.ScancodeWrapper' class to allow input handling
        :type keys: class: 'pygame.key.ScancodeWrapper'
        """
        self.forward_direction = pymunk.Vec2d(cos(self.body.angle), sin(self.body.angle))
        self.impulse = pymunk.Vec2d(1, 0) * CAR_SPEED

        if keys[self.controllerKeys[0]]:  # Move forward
            if self.reverse == 1:
                self.body.apply_impulse_at_local_point(self.impulse)
            else:
                self.body.apply_impulse_at_local_point(self.impulse * BRAKE_SPEED)
            if self.body.velocity.length < 5:
                self.reverse = 1
              
        elif keys[self.controllerKeys[1]]:  # Move backward
            if self.reverse == -1:
                self.body.apply_impulse_at_local_point(-self.impulse)
            else:
                self.body.apply_impulse_at_local_point(-self.impulse * BRAKE_SPEED)
            if self.body.velocity.length < 5:
                self.reverse = -1

        elif keys[pygame.K_SPACE]:
            if self.reverse == -1:
                self.body.apply_impulse_at_local_point(self.impulse * BRAKE_SPEED)
            elif self.reverse == 1:
                self.body.apply_impulse_at_local_point(-self.impulse * BRAKE_SPEED)
        else:
            #Simulates deceleration when no movement command is given
            self.body.velocity = (self.body.velocity.length - FREE_DECELERATION) * self.forward_direction * self.reverse
        
        #Turning the car
        self.turning_radius = CAR_SIZE[0] / sin(radians(CAR_TURN))
        if keys[self.controllerKeys[2]]:  # Turn left
            self.body.angular_velocity = self.body.velocity.length / -self.turning_radius * self.reverse
        elif keys[self.controllerKeys[3]]:  # Turn right
            self.body.angular_velocity = self.body.velocity.length / self.turning_radius * self.reverse
        else:
            self.body.angular_velocity = 0  # Stop turning when no key is pressed
        
        # Reset drift: Set velocity to only move in the direction of the car's facing angle
        self.body.velocity = self.forward_direction * self.reverse * min(self.body.velocity.length, MAX_SPEED)
        #Prints current velocity
        #print("\rVelocity:{:0.2f}".format(self.body.velocity.length),end="")

    def update(self, controls: tuple[float, float]):
        """Calculates the needed motion of the car class called. Takes tuples which will be transmitted through ros messages for controls.

        :param controls: First float controls forward throttle percentage, ranging from 1 to -1.
            Second float controls turn angle percentage, positive is right and negative is left. Ranges from 1 to -1.
        :type controls: tuple[float,float]
        """
        self.forward_direction = pymunk.Vec2d(cos(self.body.angle), sin(self.body.angle))

        self.impulse = pymunk.Vec2d(1, 0) * CAR_SPEED
        if controls[0] > 0:
            if self.reverse == 1:
                self.body.apply_impulse_at_local_point(self.impulse * controls[0])
            else:
                self.body.apply_impulse_at_local_point(self.impulse * BRAKE_SPEED)
            if self.body.velocity.length < 5:
                self.reverse = 1

        elif controls[0] < 0:  # Move backward
            if self.reverse == -1:
                self.body.apply_impulse_at_local_point(self.impulse * controls[0])
            else:
                self.body.apply_impulse_at_local_point(-self.impulse * BRAKE_SPEED)
            if self.body.velocity.length < 5:
                self.reverse = -1

        else:
            self.body.velocity = (self.body.velocity.length - FREE_DECELERATION) * self.forward_direction * self.reverse

        self.turning_radius = CAR_SIZE[0] / sin(radians(CAR_TURN))

        if controls[1] < 0:  # Turn left
            self.body.angular_velocity = self.body.velocity.length / self.turning_radius * controls[1] * self.reverse
        elif controls[1] > 0:  # Turn right
            self.body.angular_velocity = self.body.velocity.length / self.turning_radius * controls[1] * self.reverse
        else:
            self.body.angular_velocity = 0  # Stop turning when no key is pressed

        # Reset drift: Set velocity to only move in the direction of the car's facing angle
        self.body.velocity = self.forward_direction * self.reverse * min(self.body.velocity.length, MAX_SPEED)

    def getPos(self) -> pymunk.vec2d.Vec2d:
        """Returns the car's current x and y position. Individual positions can be called through getPos().x and getPos().y

        :return: List of positional coordinates in :class:`pymunk.vec2d.Vec2d` format
        :rtype: :class:`pymunk.vec2d.Vec2d`
        """
        return self.body.position

    def getVelocity(self) -> pymunk.vec2d.Vec2d:
        """Returns the car's current velocity.

        :return: Current velocity in the :class:`pymunk.vec2d.Vec2d` format
        :rtype: :class:`pymunk.vec2d.Vec2d`
        """
        return self.body.velocity

    def getAngle(self) -> float:
        """Returns the car's current angle in degrees

        :return: Current angle in degrees clockwise from east
        :rtype: float
        """
        return degrees(self.body.angle)

    def setPos(self, pos: tuple[float, float] | pymunk.vec2d.Vec2d):
        """Set's the car's position

        :param pos: A tuple or vector of the position (x, y)
        :type pos: tuple[float, float] | pymunk.vec2d.Vec2d
        """
        self.body._set_position(pos)

    def setVel(self, vel: tuple[float, float] | pymunk.vec2d.Vec2d):
        """Sets the car's velocity

        :param vel: A tuple or vector of the position (x, y)
        :type vel: tuple[float, float] | pymunk.vec2d.Vec2d
        """
        self.body._set_velocity(vel)

    def setAngle(self, angle: float):
        """Set's the car's angle

        :param angle: The angle in degrees cw from east
        :type angle: float
        """
        self.body._set_angle(angle)


class Ball:
    """Class used to define the soccer ball
    :param x: x coordinate for the starting position of the ball
    :type x: float
    :param y: y coordinate for the starting position of the ball
    :type y: float
    :param space: Contains the :class: `pymunk.space` class object for calculating physical interactions
    :type space: class: `pymunk.space`
    :param impulse: Starting impulse applied to the ball at object creation (defaults to no impulse)
    :type imlpulse: class: `pymunk.Vec2d`
    """

    def __init__(
            self,
            x: float,
            y: float,
            space: pymunk.Space,
            impulse: pymunk.Vec2d = pymunk.Vec2d(0, 0)
    ):
        """Constructor method"""
        self.inertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        self.body = pymunk.Body(BALL_MASS, self.inertia)
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.friction = BALL_FRICTION
        self.shape.elasticity = BALL_ELASTICITY
        self.shape.color = BALL_COLOR

        self.space = space
        self.space.add(self.body, self.shape)

        self.body.apply_impulse_at_local_point(impulse)

    def decelerate(self):
        """Gradually reduces the velocity of the ball to simulate friction."""
        self.body.velocity = (self.body.velocity.length - BALL_DECELERATION) * self.body.velocity.normalized()

    def getPos(self) -> pymunk.vec2d.Vec2d:
        """Returns the ball's current x and y position. Individual positions can be called through getPos().x and getPos().y
        :return: List of positional coordinates in :class:`pymunk.vec2d.Vec2d` format
        :rtype: :class:`pymunk.vec2d.Vec2d`
        """
        return self.body.position

    def getVelocity(self) -> pymunk.vec2d.Vec2d:
        """Returns the ball's current velocity
        :return: Current velocity in the :class:`pymunk.vec2d.Vec2d` format
        :rtype: :class:`pymunk.vec2d.Vec2d`
        """
        return self.body.velocity

    def setPos(self, pos: tuple[float, float] | pymunk.vec2d.Vec2d):
        """Set's the ball's position

        :param pos: A tuple or vector of the position (x, y)
        :type pos: tuple[float, float] | pymunk.vec2d.Vec2d
        """
        self.body._set_position(pos)

    def setVelocity(self, vel: tuple[float, float] | pymunk.vec2d.Vec2d):
        """Set's the ball's velocity

        :param vel: A tuple or vector of the velocity (x, y)
        :type vel: tuple[float, float] | pymunk.vec2d.Vec2d
        """
        self.body._set_velocity(vel)


class Game:
    """Gamestate object that handles simulation of physics and handling of inputs.
    :param carlist: List of tuples where each item includes [team, x position, y position, angle from east]. Angle from east is assumed 0 if blank.
        Defaults to one car on each team equidistant from the goal and field center.
    :type carlist: list[tuple[bool, float, float, float] | tuple[bool, float, float]]
    :param ballPosition: Tuple for x and y start positions of the ball. Defaults to center of the field.
    :type ballPosition: tuple[float, float]
    """

    def __init__(
            self,
            carStartList: list[tuple[bool, float, float, float] | tuple[bool, float, float]] = CAR_POS,
            ballPosition: tuple[float, float] = BALL_POS
    ):
        """Constructor method"""
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH + GOAL_DEPTH, FIELD_HEIGHT))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()

        self.leftscore = 0
        self.rightscore = 0
        self.ticks = 60
        self.ballPosition = ballPosition
        self.carStartList = carStartList
        print(len(carStartList))
        self.cars = []
        self.inputs = []
        self.exit = False

        self.gameSpace = pymunk.Space()

    def checkGoal(
            self,
            leftgoal: float,
            rightgoal: float,
            topgoal: float,
            botgoal: float
    ):
        """Checks to see if ball position is in a goal, then calculates new score and resets the field.
            If ball leaves bounds without entering a goal, reset the field without scoring points.
        :param leftgoal: Horizontal position for left goal line
        :param rightgoal: Horizontal position for right goal line
        :param topgoal: Vertical position of the top goal bounderies
        :param botgoal: Vertical position of the bottom goal boundaries
        """
        if (self.ball.getPos().x < leftgoal):
            if (self.ball.getPos().y > topgoal) and (self.ball.getPos().y < botgoal):
                self.rightscore += 1
            self.reset()
        elif (self.ball.getPos().x > rightgoal):
            if (self.ball.getPos().y > topgoal) and (self.ball.getPos().y < botgoal):
                self.leftscore += 1
            self.reset()

    def reset(self):
        """Resets field by removing ball and car objects from the field, then calling self.addObjects()"""
        for c in self.cars:
            self.gameSpace.remove(c.body, c.shape)
        self.cars = []
        self.gameSpace.remove(self.ball.body, self.ball.shape)
        #self.addObjects()
        self.addDefaultObjects()

    def addObjects(self):
        """Adds new ball and car objects to the field according to the contents of self.carStartList"""
        self.ball = Ball(self.ballPosition[0], self.ballPosition[1], self.gameSpace)
        for c in self.carStartList:  # loops through list of start cars, creates new car object for each car listed
            self.inputs.append([0, 0])  # reset's car controls
            if len(c) == 4:  # Specifies starting angle if not given
                self.cars.append(Car(c[0], c[1], c[2], self.gameSpace, c[3]))
            else:
                self.cars.append(Car(c[0], c[1], c[2], self.gameSpace, 0))

    
    def addDefaultObjects(self) -> None:
        """Adds new ball and car objects to the field according to the contents of self.carStartList"""
        self.ball = Ball(self.ballPosition[0], self.ballPosition[1], self.gameSpace)
        multiControlList = [(pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT), (pygame.K_t, pygame.K_g, pygame.K_f, pygame.K_h), (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d), (pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l), ]
        for i, c in enumerate(self.carStartList): #loops through list of start cars, creates new car object for each car listed
            self.inputs.append([0,0]) #reset's car controls
            if len(c) == 4: #Specifies starting angle if not given
                self.cars.append(Car(c[0],c[1], c[2],self.gameSpace, c[3], controllerKeys=multiControlList[i]))
            else:
                self.cars.append(Car(c[0],c[1],c[2],self.gameSpace, 0, controllerKeys=multiControlList[i]))
    
    
    
    def updateObjects(self, walls: bool, useKeys: bool):
        """Updates controls for all objects currently on the field. Also decelerates the ball.
        :param walls: Indicates whether field walls are enabled. Does not calculate goals if false.
        :type walls: bool
        :param useKeys: Toggles usage of keyboard inputs for calculations (true), or ros messages (false)
        :type useKeys: bool
        """
        if useKeys:
            self.pressed = pygame.key.get_pressed()
            for c in self.cars:
                c.keyUpdate(self.pressed)
        else:
            for i, c in enumerate(self.cars):
                try:
                    c.update(self.inputs[i])
                except:
                    c.update([0, 0])
        self.ball.decelerate()
        if walls:
            self.checkGoal(GOAL_DEPTH, FIELD_WIDTH, SIDE_WALL, SIDE_WALL + GOAL_HEIGHT)

    def run(self, visualizer: bool = True, walls: bool = True, useKeys: bool = True):
        """Main logic function to keep track of gamestate.
        :param visualizer: Toggles rendering of the simulation.
        :type visualizer: bool
        :param walls: Toggles the walls of the field on or off, no walls also disables goal checks
        :type walls: bool
        :param useKeys: Toggles between usage of keyboard inputs or programmatic control
        :type useKeys: bool
        """
        #self.addObjects()
        self.addDefaultObjects()

        # Walls in Field
        if walls:
            self.static_lines = [
                pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, 0.0), (FIELD_WIDTH, 0.0), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, FIELD_HEIGHT), (GOAL_DEPTH, FIELD_HEIGHT),
                               0.0),

                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, 0.0), (FIELD_WIDTH, SIDE_WALL), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, SIDE_WALL),
                               (FIELD_WIDTH + GOAL_DEPTH, SIDE_WALL), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH + GOAL_DEPTH, SIDE_WALL),
                               (FIELD_WIDTH + GOAL_DEPTH, GOAL_HEIGHT + SIDE_WALL), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, GOAL_HEIGHT + SIDE_WALL),
                               (FIELD_WIDTH + GOAL_DEPTH, SIDE_WALL + GOAL_HEIGHT), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, GOAL_HEIGHT + SIDE_WALL),
                               (FIELD_WIDTH, FIELD_HEIGHT), 0.0),

                pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, SIDE_WALL), (GOAL_DEPTH, 0.0), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, SIDE_WALL), (0, SIDE_WALL), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (0, SIDE_WALL), (0, GOAL_HEIGHT + SIDE_WALL), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, GOAL_HEIGHT + SIDE_WALL),
                               (0, SIDE_WALL + GOAL_HEIGHT), 0.0),
                pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, GOAL_HEIGHT + SIDE_WALL),
                               (GOAL_DEPTH, FIELD_HEIGHT), 0.0),
            ]
            for l in self.static_lines:
                l.friction = FIELD_FRICTION
                l.elasticity = FIELD_ELASTICITY
            self.gameSpace.add(*self.static_lines)

        while not self.exit:
            self.dt = self.clock.get_time() / 1000

            # Quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            self.updateObjects(walls, useKeys)

            # Drawing
            print("\rLeft: ", self.leftscore, " Right: ", self.rightscore, end="")
            if visualizer:
                pygame.display.set_caption("fps: " + "{:.2f}".format(self.clock.get_fps()))
                self.screen.fill(pygame.Color("white"))
                self.gameSpace.debug_draw(self.draw_options)
                pygame.display.update()
            else:
                print(" | fps: " + "{:.2f}".format(self.clock.get_fps()), end="")

            self.gameSpace.step(self.dt)
            self.clock.tick(self.ticks)
        pygame.quit()


def main(args=None):
    game = Game()
    game.run(visualizer=True, walls=True, useKeys=True)


if __name__ == "__main__":
    main()
