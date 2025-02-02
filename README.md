# Rocket League 2

## Simulator

### Game Class

This is the gamestate object that handles simulation of physics and handling of inputs.

Methods:

`checkGoal(leftGoal, rightGoal, topGoal, botGoal)` : Checks to see if ball position is in a goal, then calculates new score and resets the field. If ball leaves bounds without entering a goal, reset the field without scoring points.

`reset()` : Resets field by removing ball and car objects from the field, then calling `self.addObjects()`

`addObjects()` : Adds new ball and car objects to the field according to the contents of self.carStartList

`updateObjects(walls, useKeys)` : Updates controls for all objects currently on the field. Also decelerates the ball. Takes in 2 booleans if it should consider walls and if it should use keyboard input

`handleInputs(msg)` : Internal method to handle ROS messages

`broadcast()` : Broadcasts a ROS message containing ball and car positions

`run(visualizer, walls, useKeys)` : Main logic function to keep track of gamestate. Takes input from ROS messages. Three inputs: visualizer (renders the game), walls (whether to consider walls or not) and useKeys (whether to take keyboard input or not)

### Car Class

This is the class used to define each car in the simulator.

Methods:

`keyUpdate(keys)` : Calculates the needed motion of the car class called. Takes user keyboard inputs for controls

`update(controls)` : Calculates the needed motion of the car class called. Takes tuples which will be transmitted through ros messages for controls.

`getPos()` : Returns current x and y positions as a 2D vector. Individual components can be called using getPos().x and getPos().y

`getVelocity()` : Returns the car's current velocity as a 2D Vector. Individual components can be called using getPos().x and getPos().y

`getAngle()` : Returns the car's current angle in degrees

### Ball Class

This is the class used to define the soccer ball.

Methods:

`decelerate()` : Gradually reduces the velocity of the ball to simulate friction.

`getPos()` : Returns current x and y positions as a 2D vector. Individual components can be called using getPos().x and getPos().y

`getVelocity()` : Returns the ball's current velocity as a 2D Vector. Individual components can be called using getPos().x and getPos().y
