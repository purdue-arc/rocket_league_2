import rclpy
import rclpy.logging
from rktl_interfaces.msg import CarAction, Field, Pose


def callbackCarAction(cAction):
    print(f"recieved: \nCarAction.id: {cAction.id}")
    print(f"CarAction.throttle: {cAction.throttle}")
    print(f"CarAction.steer: {cAction.steer}")
def callbackField(fld):
    print(f"recieved: \nField.ball_pose: {fld.ball_pose}")
    print(f"Field.team1_poses: {fld.team1_poses}")
    print(f"Field.team2_poses: {fld.team2_poses}")

def listener():
    rclpy.init_node()