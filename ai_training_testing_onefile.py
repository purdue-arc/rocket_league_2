from queue import Queue
from event import Message
from stable_baselines3 import A2C, PPO

import TouchBall1


class AITesting():
    # TODO: typing for model_path and env inputs (IDK what data types they are)
    def __init__(
        self,
        model_path,
        env,
        inQueue: Queue,
        outQueue: Queue
    ):
        """Constructor Method

        :param model_path: _description_
        :type model_path: _type_
        :param env: _description_
        :type env: _type_
        :param inQueue: Queue for messages into the AI
        :type inQueue: Queue
        :param outQueue: Queue for messages out of the AI
        :type outQueue: Queue
        """
        self.model = PPO.load(model_path, env=env)
        self.inQueue = inQueue
        self.outQueue = outQueue
    
    # TODO: I dunno what this function does so I kept it in
    def recieved_field_data(self, field):
        action, _state = self.model.predict(field)
        return action, _state
    
    def run(self):
        """Main run method, recieves queue messages and sends actions
        """
        while True:
            data = self.inQueue.get()
            if data.getType() == 'exit':
                self.outQueue.put(Message("exit", None))
                break
            if data.getType() == 'fieldState':
                action, _state = self.model.predict(data.getData())
                """TODO: Turn 'action' object into a dict object to be sent via Message object
                Should use this formatting if possible:
                {
                    "throttle": THROTTLE VALUE FROM -1 TO 1,
                    "steer": STEERING VALUE FROM -1 TO 1
                }
                """
                self.outQueue.put(Message('controlAction', action))
            
            # TODO: Should data be put back into the queue? I'm not sure, but I doubt it.
            else:
                self.inQueue.put(data)

class AITraining():
    def __init__(self, inQueue: Queue, outQueue: Queue, killobject: object):
        env = TouchBall1()
        self.model = A2C("MultiInputPolicy", env, verbose=1, device='cuda')
        self.killobject = killobject
        self.inQueue = inQueue
        self.outQueue = outQueue
    
    def beginLearn(self):
        self.model.learn(100000)
    def recieved_field_data(self, field):
        self.field = field
    def callback_get_obs(self):
        return self.field
    