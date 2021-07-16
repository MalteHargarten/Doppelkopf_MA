import numpy as np
from typing import List
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.data.DataCache import DataCache
from doppelkopf.data.Experience import Experience
from doppelkopf.data.GameSequence import GameSequence

class ReplayBuffer():
    def __init__(self, writeIndex=0, cacheSize=0, loadStart=0):
        self.writeIndex = writeIndex
        self.cache = DataCache(cacheSize)
        self.loadStart = loadStart

    def ResetLoadStart(self):
        self.loadStart = 0

    def ResetWriteIndex(self):
        self.nextIndex = 0
    
    def ClearCache(self):
        self.cache.Clear()

    def ClearAll(self):
        self.ResetWriteIndex()
        self.ResetLoadStart()
        self.ClearCache()

    def Size(self):
        return self.cache.Size()

    def IsFull(self):
        return self.cache.IsFull()

    def PutNext(self, experience: Experience, replace=False) -> bool:
        nextKey = str(self.writeIndex)
        self.writeIndex += 1
        return self.cache.Put(nextKey, experience, replace)

    def Put(self, key, experience: Experience, replace=False) -> bool:
        return self.cache.Put(key, experience, replace)

    '''
    def PutNext(self, sequence: GameSequence, replace=False) -> bool:
        nextKey = str(self.writeIndex)
        self.writeIndex += 1
        return self.cache.Put(nextKey, sequence, replace)

    def Put(self, key, sequence: GameSequence, replace=False) -> bool:
        return self.cache.Put(key, sequence, replace)

    def GetSequenceAt(self, sequenceIndex: int) -> GameSequence:
        return self.GetValue(str(sequenceIndex))

    def GetValue(self, key: str) -> GameSequence:
        return self.cache.Get(key)
    '''

    def GetExperienceAt(self, index: int) -> Experience:
        return self.GetValue(str(index))

    def GetValue(self, key: str) -> Experience:
        return self.cache.Get(key)

    def HasNextLoad(self) -> bool:
        return self.loadStart < self.Size()

    '''
    def NextLoad(self, loadSize: int, shuffle: bool) -> list:
        start = self.loadStart
        stop = min(start + loadSize, self.Size()) # Either take as many elements as "batchSize" or the last element we actually have
        if start == stop:
            return self.GetSequenceAt(start)
        inverse = stop < start
        howmany = (stop - start) if not inverse else (start - stop)
        sequences = []
        for i in range(howmany):
            sequences.append(self.GetSequenceAt(i))
        if shuffle:
            np.random.shuffle(sequences)
        self.loadStart += howmany
        return sequences

    def RandomSequences(self, numOfSequences):
        indices = list(self.cache.data.keys())
        numOfSequences = min(numOfSequences, len(indices)) # If there aren't enough entries in the cache to begin with, only use those, that are available
        np.random.shuffle(indices) # in-place shuffle
        indices = indices[:numOfSequences] # Get the first indices from the randomly shuffled list
        result = []
        for index in indices:
            result.append(self.GetSequenceAt(index))
        return result
    '''

    def NextLoad(self, loadSize: int, shuffle: bool) -> List[Experience]:
        start = self.loadStart
        stop = min(start + loadSize, self.Size()) # Either take as many elements as "batchSize" or the last element we actually have
        if start == stop:
            return self.GetExperienceAt(start)
        inverse = stop < start
        howmany = (stop - start) if not inverse else (start - stop)
        experiences = []
        for i in range(howmany):
            experiences.append(self.GetExperienceAt(i))
        if shuffle:
            np.random.shuffle(experiences)
        self.loadStart += howmany
        return experiences

    def RandomBatch(self, batchSize: int) -> List[Experience]:
        indices = list(self.cache.data.keys())
        numOfSequences = min(batchSize, len(indices)) # If there aren't enough entries in the cache to begin with, only use those, that are available
        np.random.shuffle(indices) # in-place shuffle
        indices = indices[:numOfSequences] # Get the first indices from the randomly shuffled list
        result = []
        for index in indices:
            result.append(self.GetExperienceAt(index))
        return result