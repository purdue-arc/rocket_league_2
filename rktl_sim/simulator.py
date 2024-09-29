import pygame
import pymunk.pygame_util
import pymunk
from math import sin, radians, cos

#Car Specs
CAR_SIZE = (16.5, 8.5) # (Length, Width)
CAR_MASS = 5
CAR_SPEED = 5  # Impulse applied for forward/backward movement
CAR_TURN = 30  # Angular velocity for car turning
BRAKE_SPEED = 5  # Multiplier of CAR_SPEED for braking force
FREE_DECELERATION = 0.5  # Rate velocity decreases with no input
CAR_FRICTION = 0.5
CAR_COLOR = pygame.Color("green")

#Field Specs
FIELD_WIDTH = 426.72
FIELD_HEIGHT = 304.8
FIELD_FRICTION = 0.3
FIELD_ELASTICITY = 0.5
GOAL_HEIGHT = 81.28
GOAL_DEPTH = 25.4
FIELD_COLOR = pygame.Color("white")

#Ball Specs
BALL_MASS = 0.1
BALL_RADIUS = 6.85 / 2
BALL_POS = (FIELD_WIDTH + GOAL_DEPTH) / 2, FIELD_HEIGHT / 2 # Starting position of the ball (x, y)
BALL_ELASTICITY = 1
BALL_FRICTION = 0.5
BALL_COLOR = pygame.Color("blue")

#Sim Limits
MAX_SPEED = 200 # Max speed limit of the cars

class Car:
    def __init__(self, x, y, space, angle=0):
        self.body = pymunk.Body(CAR_MASS, pymunk.moment_for_box(CAR_MASS, CAR_SIZE))
        self.body.position = (x, y)
        self.body.angle = radians(angle)
        self.shape = pymunk.Poly.create_box(self.body, CAR_SIZE)
        self.shape.color = CAR_COLOR
        self.shape.friction = CAR_FRICTION
        self.space = space
        self.space.add(self.body, self.shape)

        self.steering = 0.0 # Rate of steering
        self.reverse = 1 # Stores which direction car moves, 1 for forwards and -1 for backwards

    def update(self, keys, dt):
        self.forward_direction = pymunk.Vec2d(cos(self.body.angle), sin(self.body.angle))
        
        self.impulse = pymunk.Vec2d(1, 0) * CAR_SPEED
        if keys[pygame.K_UP]:  # Move forward
            if self.reverse == 1:
                self.body.apply_impulse_at_local_point(self.impulse)
            else:
                self.body.apply_impulse_at_local_point(self.impulse * BRAKE_SPEED)
            if self.body.velocity.length < 5:
                self.reverse = 1
              
        elif keys[pygame.K_DOWN]:  # Move backward
            #self.body.velocity -= self.forward_direction
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
        
        # Turning the car
        self.turning_radius = CAR_SIZE[0] / sin(radians(CAR_TURN))
        if keys[pygame.K_LEFT]:  # Turn left
            self.body.angular_velocity = self.body.velocity.length / -self.turning_radius * self.reverse
        elif keys[pygame.K_RIGHT]:  # Turn right
            self.body.angular_velocity = self.body.velocity.length / self.turning_radius * self.reverse
        else:
            self.body.angular_velocity = 0  # Stop turning when no key is pressed
        
        # Reset drift: Set velocity to only move in the direction of the car's facing angle
        self.body.velocity = self.forward_direction * self.reverse * min(self.body.velocity.length, MAX_SPEED)
        #Prints current velocity
        print("\rVelocity:{:0.2f}".format(self.body.velocity.length),end="")

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH + GOAL_DEPTH, FIELD_HEIGHT))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.gameSpace = pymunk.Space()

    def run(self):
        car1 = Car((FIELD_WIDTH + GOAL_DEPTH) / 3, FIELD_HEIGHT / 2, self.gameSpace)
        car2 = Car(2 * (FIELD_WIDTH + GOAL_DEPTH) / 3, FIELD_HEIGHT / 2, self.gameSpace, 180)

        # Walls in Field
        sideWallLength = (FIELD_HEIGHT - GOAL_HEIGHT) / 2
        static_lines = [
            pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, 0.0), (FIELD_WIDTH, 0.0), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, FIELD_HEIGHT), (GOAL_DEPTH, FIELD_HEIGHT), 0.0),

            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, 0.0), (FIELD_WIDTH, sideWallLength), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, sideWallLength), (FIELD_WIDTH + GOAL_DEPTH, sideWallLength), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH + GOAL_DEPTH, sideWallLength), (FIELD_WIDTH + GOAL_DEPTH, GOAL_HEIGHT + sideWallLength), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, GOAL_HEIGHT + sideWallLength), (FIELD_WIDTH + GOAL_DEPTH, sideWallLength + GOAL_HEIGHT), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (FIELD_WIDTH, GOAL_HEIGHT + sideWallLength), (FIELD_WIDTH, FIELD_HEIGHT), 0.0),

            pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, sideWallLength), (GOAL_DEPTH, 0.0), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, sideWallLength), (0, sideWallLength), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (0, sideWallLength), (0, GOAL_HEIGHT + sideWallLength), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, GOAL_HEIGHT + sideWallLength), (0, sideWallLength + GOAL_HEIGHT), 0.0),
            pymunk.Segment(self.gameSpace.static_body, (GOAL_DEPTH, GOAL_HEIGHT + sideWallLength), (GOAL_DEPTH, FIELD_HEIGHT), 0.0),
        ]
        for l in static_lines:
            l.friction = FIELD_FRICTION
            l.elasticity = FIELD_ELASTICITY
        self.gameSpace.add(*static_lines)
        
        self.inertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, (0, 0))
        self.ball = pymunk.Body(BALL_MASS, self.inertia)
        self.ball.position = BALL_POS
        self.ball_shape = pymunk.Circle(self.ball, BALL_RADIUS, (0, 0))
        self.ball_shape.friction = BALL_FRICTION
        self.ball_shape.elasticity = BALL_ELASTICITY
        self.ball_shape.color = BALL_COLOR
        self.gameSpace.add(self.ball, self.ball_shape)

        while not self.exit:
            self.dt = self.clock.get_time() / 1000

            # Quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()

            # Logic
            car1.update(pressed, self.dt)
            car2.update(pressed, self.dt)

            # Drawing
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
            self.screen.fill(pygame.Color("white"))
            self.gameSpace.debug_draw(self.draw_options)
            pygame.display.update()
            self.gameSpace.step(self.dt)

            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
