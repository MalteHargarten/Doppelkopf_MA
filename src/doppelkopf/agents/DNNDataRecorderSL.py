from doppelkopf.utils.Console import Console
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.game.Card import Card
from doppelkopf.game.Client import Client
from doppelkopf.data.DNNDataset import DNNDataset
from doppelkopf.game.GameState import GameState
from doppelkopf.agents.RulebasedPlayer import RulebasedPlayer

class DNNDataRecorderSL(RulebasedPlayer):
    def __init__(self, datasetFilepath):
        super(DNNDataRecorderSL, self).__init__() # Call base constructor
        self.datasetFilepath = datasetFilepath
        self.dataset = DNNDataset(self.datasetFilepath)

    def instantiateClient(self) -> Client:
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onStateReceived=self.receiveGameState
        )

    def PlayGames(self, numOfGames):
        opened = self.dataset.Open()
        success = super(DNNDataRecorderSL, self).PlayGames(numOfGames) # Call base method
        if opened:
            self.dataset.Close()
        return success

    def receiveGameState(self, state: GameState, isFinalState: bool):
        if not isFinalState: # Final states (the last states of each trick/game) are not used for training, because no card is to be picked in this situation anyway
            labels = Doppelkopf.getLegalPlayableMask(state.currentStack[0] if len(state.currentStack) > 0 else None, self.client.myPlayer)
            self.dataset.StoreStateLabel(state, labels)