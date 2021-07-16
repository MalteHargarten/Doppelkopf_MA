import numpy as np
from typing import Tuple
from doppelkopf.utils.Console import Console
from doppelkopf.data.Dataset import Dataset
from doppelkopf.game.CardFeedback import CardFeedback
from doppelkopf.game.Team import Team
from doppelkopf.game.Card import Card
from doppelkopf.game.Player import Player
from doppelkopf.game.GameState import GameState

class DNNDataset(Dataset):
    KEY_STATE = "state"
    KEY_LABEL = "label"

    def __init__(self, filename, nextIndex=None, cacheSize=0, loadStart=0, maxSize=None):
        super(DNNDataset, self).__init__(filename, nextIndex, cacheSize, loadStart, maxSize) # Call base Constructor

    def StoreStateLabel(self, state: GameState, label: np.ndarray):
        value = { DNNDataset.KEY_STATE: state.Flat(), DNNDataset.KEY_LABEL: label } # Store state and label as a dict in the shelve file
        self.PutNext(value)

    def GetStateLabelAt(self, index: int, cacheResult) -> tuple:
        value = self.GetValue(str(index), cacheResult)
        if value is not None:
            state = value[DNNDataset.KEY_STATE]
            label = value[DNNDataset.KEY_LABEL]
            return state, label
        return value # If value is None, return it

    def NextLoad(self, loadSize: int, shuffle: bool, cacheResult: bool) -> Tuple[np.ndarray, np.ndarray]:
        opened = self.Open()
        Console.WriteDebug("Begin NextLoad(%d)" % (loadSize))
        start = self.loadStart
        stop = min(start + loadSize, self.Size()) # Either take as many elements as "batchSize" or the last element we actually have
        if start == stop:
            return self.GetStateLabelAt(start, cacheResult)
        inverse = stop < start
        howmany = (stop - start) if not inverse else (start - stop)
        x = np.zeros(shape=(howmany, GameState.SIZE_STATE))
        y = np.zeros(shape=(howmany, Card.NUM_CARDTYPES))
        Console.WriteInfo("0/%d loaded" % (howmany))
        for i in range(howmany):
            state, label = self.GetStateLabelAt(start + i, cacheResult)
            x[i] = state
            y[i] = label
            Console.WriteOverPreviousLine("%d/%d loaded" % ((i+1), howmany))
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(x)
            np.random.set_state(rng_state)
            np.random.shuffle(y)
        self.loadStart += howmany
        if opened:
            self.Close()
        return x, y

    def VerifyEntry(self, stateFlat, label) -> bool:
        isValid = True
        index = 0
        player = Player(0)
        teamRe, teamKontra = Team.createTeams()
        state = GameState.FromFlat(stateFlat, index)
        player.handCards(state.playerDeck, teamRe, teamKontra) # Give cards to player
        for card in Card.CARDTYPES:
            feedback = player.getFeedback(card.cardType, state.currentStack[0] if len(state.currentStack) > 0 else None)
            if feedback == CardFeedback.NOT_ALLOWED or feedback == CardFeedback.NOT_IN_HAND:
                if label[card.cardType] != 0.0:
                    Console.WriteError("Found a cardType labeled as %f instead of 0.0" % (label[card.cardType]))
                    isValid = False
            elif feedback == CardFeedback.OK or feedback == CardFeedback.OK_COULD_NOT_FOLLOW_SUIT:
                if label[card.cardType] != 1.0:
                    Console.WriteError("Found a cardType labeled as %f instead of 1.0" % (label[card.cardType]))
                    isValid = False
        return isValid

    def Verify(self):
        self.Open()
        loadSize = self.Size()
        while self.HasNextLoad():
            x,y = self.NextLoad(loadSize, shuffle=True, cacheResult=False)
            for i in range(loadSize):
                if self.VerifyEntry(x[i], y[i]):
                    Console.WriteSuccess("%d is valid" % (i))
        self.Close()