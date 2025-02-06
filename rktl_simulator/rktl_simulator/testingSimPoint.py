import rclpy
from rktl_interfaces.msg import CarAction, Field
from simulatorPoint import PointGame
import threading
from simulator import FIELD_WIDTH, GOAL_DEPTH, FIELD_HEIGHT

def printFieldState(msg):
    print("recieved:")
    print(f"Ball Pose: x:{msg.ball_pose.x}, y:{msg.ball_pose.y}, id:{msg.ball_pose.id}")
    print(f"Car Pose: x:{msg.team1_poses[0], }")

def broadcast():
    print("sending message")
    send = CarAction()
    send.id = 0
    send.throttle = 1
    send.steer = 1
    pub.publish(send)



game = PointGame(carStartList=[
    [True, (FIELD_WIDTH + GOAL_DEPTH) / 3,FIELD_HEIGHT / 2]
])

gameThread = threading.Thread(target=game.run)

localNode = rclpy.create_node("tester")
sub = localNode.create_subscription(Field, "simTopic", printFieldState, 10)
pub = localNode.create_publisher(CarAction, "aiTopic", 10)

gameThread.start()
broadcast()
while True:
    pass