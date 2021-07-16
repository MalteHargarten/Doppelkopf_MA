from doppelkopf.game.GameState import GameState

class Experience():
    def __init__(self, state: GameState, action: int, reward: float, nextState: GameState):
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState