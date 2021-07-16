from enum import Enum

class CardFeedback(Enum):
    NOT_IN_HAND = 0
    NOT_ALLOWED = 1
    OK = 2
    OK_COULD_NOT_FOLLOW_SUIT = 3