import numpy as np
from doppelkopf.game.Client import Client
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.agents.RulebasedPlayer import RulebasedPlayer
from doppelkopf.data.LSTMSupervisedDataset import LSTMSupervisedDataset

class LSTMSupervisedRecorder(RulebasedPlayer):
    def __init__(self, filepath, cacheSize=0):
        super(LSTMSupervisedRecorder, self).__init__()
        self.dataset = LSTMSupervisedDataset(filepath, cacheSize=cacheSize)
        self.gameStates = []
        self.labels = []

    def instantiateClient(self) -> Client:
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onStateReceived=self.receiveGameState,
            onGameCompleted=self.onGameCompleted,
        )

    def receiveGameState(self, state: GameState, isFinalState: bool):
        self.gameStates.append(state.Flat())
        self.labels.append(Doppelkopf.getLegalPlayableMask(state.currentStack[0] if len(state.currentStack) > 0 else None, self.client.myPlayer))

    def onGameCompleted(self, winnerTeamName: str, score: int):
        states = np.array(self.gameStates)
        labels = np.array(self.labels)
        self.dataset.PutNext({"states": states, "labels": labels})
        self.gameStates.clear()
        self.labels.clear()