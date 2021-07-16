from enum import Enum
class PickCardMode(Enum):
    EXPLORATION=0,
    EXPLOITATION=1

class RewardMode(Enum):
    PER_CARD=0
    PER_TRICK=1
    PER_GAME=2

class RewardType(Enum):
    PER_VALID_CARD=0
    PER_TRICK_SIMPLE=1
    PER_TRICK_PROPORTIONAL=2
    PER_TRICK_PROPORTIONAL_FIXED_RATE=3
    PER_GAME_SIMPLE=4
    PER_GAME_PROPORTIONAL=5
    PER_GAME_PROPORTIONAL_FIXED_RATE=6

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return RewardType(int(value))
        elif isinstance(value, int):
            return RewardType(value)
        return super()._missing_(value)

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)