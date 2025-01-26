# Rocket League 2 Without ROS
## Main Files Are:
- `ai_training_testing_onefile.py`
- `event.py`
- `main.py`
- `perception.py (not updated)`
- `simulator.py`
- `TouchBall1.py (not updated)`
## Event.py:
Contains `Message` Class
A Message consists of a message type (string) and message data (dict)
`Message.MESSAGE_TYPES` is a list of valid types
## Simulator.py
Added `runWithQueue` method to `Game` class
Takes in two queue objects (in and out queues)
Can accept messages from perception to update ball position and multiple car positions
Can accept input messages from AI (throttle and steering)
Sends out simFieldState messages containing information about:
1. Car Positions
2. Car Velocities
3. Car Angles
4. Ball Position
5. Ball Velocity
6. dt
7. time
8. walls
Not sure if timing is set up correctly yet
