class Message:
    class InvalidEventTypeError(Exception):
        def __init__(self, event_type, valid_event_types):
            super().__init__(f"Invalid event type: '{event_type}'.\
                Valid Event Types are: {', '.join(valid_event_types)}")
    
    MESSAGE_TYPES = [
        # Universal exit message:
        'exit',
        # Update position (perception):
        'ballPosition',
        'carPosition',
        'carPositionIndexed',
        # To handle multiple cars and positions being updated:
        'carPositions',
        # AI Control action:
        'controlAction',
        'controlActionIndexed',
        # Field state from simulator
        'simFieldState'
    ]
    def __init__(self, messageType: str, messageData: dict = None):
        if messageType in self.MESSAGE_TYPES:
            self.messageType = messageType
            self.messageData = messageData
        else:
            raise Message.InvalidEventTypeError(messageType, self.MESSAGE_TYPES)
    
    def getData(self) -> dict:
        return self.messageData
    
    def getType(self) -> str:
        return self.messageType

if __name__ == "__main__":
    m = Message('ballPosition', {"x": 0.001, 'y': 5.8})
    m.getData().get()