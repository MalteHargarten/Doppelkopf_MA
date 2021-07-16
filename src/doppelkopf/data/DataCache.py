from collections import deque
from sys import maxsize

class DataCache():
    def __init__(self, maxSize=0):
        self.data = {}
        self.keyQueue = deque() # Always append() on the left and pop() on the right
        self.maxSize = maxSize
        self.currentIterIndex = 0

    #region Iterable Protocol
    def __iter__(self):
        self.currentIterIndex = 0
        return self

    def __next__(self):
        if self.currentIterIndex < self.Size():
            key = list(self.data.keys())[self.currentIterIndex] # Get key at index 'currentIterIndex'
            self.currentIterIndex += 1
            return self.data[key]
        else:
            raise StopIteration
    #endregion

    def Clear(self):
        self.data.clear()
        self.keyQueue.clear()

    def hasRoom(self):
        return self.maxSize > len(self.data)

    def Put(self, key, value, replace=False):
        if not key in self.data.keys(): # If the key doesn't already exist, add it to the dictionary and the queue
            if not self.hasRoom(): # If the Cache is full
                self.DeleteOldest()
            self.data[key] = value
            self.keyQueue.appendleft(key) # append() on the left, pop() on the right
            return True
        elif replace: # If the key already exists and replace is True, update the dictionary without updating the queue
            self.data[key] = value
            return True
        return False

    def Delete(self, key):
        if key in self.keyQueue and key in self.data:
            del self.data[key]
            self.keyQueue.remove(key) # Remove the first occurence of 'key'
            return True
        return False

    def DeleteOldest(self):
        if self.Size() > 0:
            oldestKey = self.keyQueue.pop() # pop() = popRight()
            del self.data[oldestKey]
            return True
        return False

    def Get(self, key):
        return self.data[key]

    def CheckLengths(self):
        if len(self.data) != len(self.keyQueue):
            raise ValueError("data and keyQueue have different lengths: data (%d) vs. keyQueue(%d)" % (len(self.data), len(self.keyQueue)))

    def Size(self):
        return len(self.data)

    def IsFull(self):
        return self.Size() == self.maxSize