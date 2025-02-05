import rclpy
from simulatorPoint import PointGame
from rktl_interfaces.msg import Field, Pose
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
