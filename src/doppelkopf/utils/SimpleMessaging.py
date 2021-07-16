import pickle
from enum import Enum
from doppelkopf.utils.Console import Console

class Ping_Pong_Type(Enum):
    PING = 0
    PONG = 1
    PING_PONG = 2

class MessageType(Enum):
    DISCONNECT = 0
    READY_TO_PLAY = 1
    NOT_YET_READY = 2
    PING = 3
    PONG = 4
    PLAYER_OBJECT = 5
    PLAYERLIST = 6
    TEAMS = 7
    PLAYER_HAND = 8
    GAME_STATE = 9
    CHOSEN_CARD = 10
    CARDFEEDBACK = 11
    TRICK_COMPLETED = 12
    GAME_COMPLETED = 13
    WAS_QUEEN = 14

    def to_int(self):
        return self.value

class MessageStatus(Enum):
    CREATED = 0
    QUEUED = 1
    SENT = 2
    FAILED = 3

    def to_int(self):
        return self.value

class Message():
    NEGOTIATION_MSG_SIZE = 4 # Allows the message size to be a maximum of 2,147,483,647 bytes (2.14 Gigabytes). If that's not enough, use a value of 8 (would allow for 9,223,372.03 Terabytes)
    BYTE_ORDER = 'big'
    
    def __init__(self, messageType: MessageType, data):
        self.messageType = messageType
        self.data = data
        self.status = MessageStatus.CREATED

    def Set_Data(self, newData):
        self.data = newData

    def GetData(self):
        return self.data

    def Set_Status(self, newStatus):
        self.status = newStatus

    def Get_Status(self):
        return self.status

    def __str__(self):
        return "[" + str(self.messageType) + " | " + str(self.data) + "]"

    def __repr__(self):
        return self.__str__()

Message.PING = Message(MessageType.PING, "Ping x")
Message.PONG = Message(MessageType.PONG, "Pong x")
Message.DISCONNECT = Message(MessageType.DISCONNECT, "Stop")
Message.READY_TO_PLAY = Message(MessageType.READY_TO_PLAY, "Ready")
Message.NOT_YET_READY = Message(MessageType.NOT_YET_READY, "Not yet ready")

# Provides simple wrappers for sending and receiving messages, without need of instantiation
class SimpleMessaging():
    @staticmethod
    def SendMessage(socket, message: Message):
        try:
            #socket.settimeout(timeout)            
            message_bytes = pickle.dumps(message) # Picke the message to turn it into a bytes-object
            #msg = struct.pack('>I', len(message)) + message # Each message is prefixed with its own length
            msg = bytearray()
            msg.extend(len(message_bytes).to_bytes(Message.NEGOTIATION_MSG_SIZE, Message.BYTE_ORDER)) # Add the length of the pickled message to 'msg'
            msg.extend(message_bytes) # Add the actual pickled message to 'msg'
            socket.sendall(msg) # sendall() blocks until all bytes are sent or an error is thrown
            message.Set_Status(MessageStatus.SENT)
            return None
        except Exception as error:
            message.Set_Status(MessageStatus.FAILED)
            Console.WriteError("Encountered Error while sending: %s\nError Message: %s" % (message, error), "SimpleMessaging.SendMessage()")
            return error

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @staticmethod
    def recvall(socket, n_bytes):
        data = bytearray()
        while len(data) < n_bytes:
            packet = socket.recv(n_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    @staticmethod
    def ReceiveMessage(socket) -> Message:
        try:
            #socket.settimeout(timeout)            
            raw_length = SimpleMessaging.recvall(socket, Message.NEGOTIATION_MSG_SIZE)
            if not raw_length:
                return None
            #message_length = struct.unpack('>I', raw_length)[0]
            message_length = int.from_bytes(raw_length, Message.BYTE_ORDER)
            message = SimpleMessaging.recvall(socket, message_length)
            message = pickle.loads(message) # Unpickle the bytes-object to get the original message object
            return message
        except Exception as error:
            Console.WriteError("Encountered Error while receiving\nError Message: %s" % error, "SimpleMessaging.ReceiveMessage()")
            return None

    @staticmethod
    def ReceiveMessageData(socket, expectedType):
        message = SimpleMessaging.ReceiveMessage(socket)
        if message is not None:
            if message.messageType == expectedType:
                return message.GetData()
            else:
                Console.WriteError("Expected a message of type %s, but got %s instead" % (expectedType, message.messageType), "SimpleMessaging.ReceiveMessageData()")
                raise ValueError("Expected a message of type %s, but got %s instead" % (expectedType, message.messageType))
        else:
            Console.WriteError("Expected a message of type %s, but got None" % (expectedType), "SimpleMessaging.ReceiveMessageData()")
            raise ValueError("Expected a message of type %s, but got None"  % (expectedType))