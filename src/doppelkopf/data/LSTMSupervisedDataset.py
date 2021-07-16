import numpy as np
from doppelkopf.game.Card import Card
from doppelkopf.data.Dataset import Dataset
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf

class LSTMSupervisedDataset(Dataset):
    def __init__(self, filename: str, nextIndex=0, cacheSize=0, loadStart=0, maxSize=None):
        super(LSTMSupervisedDataset, self).__init__(filename, nextIndex=nextIndex, cacheSize=cacheSize, loadStart=loadStart, maxSize=maxSize)

    def GetStatesAndLabels(self, index: int, cacheResult=False):
        value = self.GetValue(str(index), cacheResult)
        states = value['states']
        labels = value['labels']
        return states, labels

    def NextLoad(self, loadSize: int, shuffle: bool, cacheResult: bool):
        opened = self.Open()
        Console.WriteDebug("Begin NextLoad(%d)" % (loadSize))
        start = self.loadStart
        stop = min(start + loadSize, self.Size()) # Either take as many elements as "batchSize" or the last element we actually have
        if start == stop:
            return self.GetValue(str(start), cacheResult)
        inverse = stop < start
        howmany = (stop - start) if not inverse else (start - stop)
        x = np.zeros(shape=(howmany, Doppelkopf.MAX_STATES_PER_GAME, GameState.SIZE_STATE))
        y = np.zeros(shape=(howmany, Doppelkopf.MAX_STATES_PER_GAME, Card.NUM_CARDTYPES))
        Console.WriteInfo("0/%d loaded" % (howmany))
        for i in range(howmany):
            states, labels = self.GetStatesAndLabels(start + i, cacheResult)
            x[i] = states
            y[i] = labels
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