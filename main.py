from ai_training_testing_onefile import AITesting, AITraining
from simulator import Game
import perception
import threading
from queue import Queue

def main():
    g = Game()
    """For each car, run a different AI.
    Also, don't forget to pass in different queues, otherwise
    inputs will get mixed up between cars and AI.
    """
    qInAI = Queue()
    qOutAI = Queue()
    
    # TODO: Update AI initialization once AI has been fully implemented
    ai = AITesting(None, None, qInAI, qOutAI)
    
    # g.run(visualizer=True, walls=True, useKeys=True)
    gameThread = threading.Thread(target=g.runWithQueue, args = (qOutAI, qInAI, True))
    aiThread = threading.Thread(target=ai.run, args = ())
    
    gameThread.start()
    aiThread.start()
    
    gameThread.join()
    aiThread.join()
    