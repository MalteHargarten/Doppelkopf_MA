import os
import shelve
from abc import ABC, abstractmethod
from doppelkopf.utils.File import File
from doppelkopf.data.DataCache import DataCache
from doppelkopf.utils.Console import Console

# Make class abstract by inheriting from ABC (abc module)
class Dataset(ABC):
    KEY_SIZE = "size"

    def __init__(self, filename: str, nextIndex=None, cacheSize=0, loadStart=0, maxSize=None):
        self.filename = filename
        self.nextIndex = nextIndex
        self.loadStart = loadStart
        self.maxSize = maxSize
        self.isOpen = False
        self.cache = DataCache(cacheSize)
        self.d = None # 'd' is the dictionary-like shelve object
        self.size = self.Size() # Try reading 'size' from file
        if self.size is None: # If file does not contain a key for 'size'
            self.size = 0
            self.WriteSize()
        if self.nextIndex is None:
            self.nextIndex = self.size

    def IncrementSize(self):
        self.size += 1 # Increment size counter
        self.WriteSize() # Since we just increased the size, write the new size to file

    def DecrementSize(self):
        self.size -= 1 # Decrement size counter
        self.WriteSize() # Since we just decreased the size, write the new size to file

    #region Clear and Reset
    def ResetSize(self):
        self.size = 0
        self.WriteSize()

    def ResetLoadStart(self):
        self.loadStart = 0

    def ResetWriteIndex(self):
        self.nextIndex = 0
    
    def ClearCache(self):
        self.cache.Clear()

    def ClearFile(self): # Delete the files and recreate them by opening a new shelve
        self.Close() # Close if it's currently opened
        fileEndings = [".dat", ".bak", ".dir"]
        for fileEnding in fileEndings:
            File.Delete(self.filename + fileEnding)
        # Open a new shelve, thus creating new files
        self.Open()
        self.Close() # Close so as to not leave an unused open connection
        self.ResetSize()

    def ClearAll(self):
        self.ResetWriteIndex()
        self.ResetLoadStart()
        self.ClearCache()
        self.ClearFile()
    #endregion

    #region Put
    def PutNext(self, value):
        replace = False
        if self.maxSize is not None and self.size >= self.maxSize: # If Dataset has reached its maximum index
            self.nextIndex = 0 # Start back at zero
            replace = True # If maxSize is reached, start overwriting existing entries. Otherwise, add new
        self.Put(str(self.nextIndex), value, replace)
        self.nextIndex += 1

    def Put(self, key: str, value, replace):
        opened = self.Open()
        if key in self.d and replace: # If 'key' exists on file AND replace is True
            self.Delete(key) # Delete key (decrements the counter)
        if self.maxSize is None or self.size < self.maxSize: # Only if there is capacity left
            self.d[key] = value # Store key-value pair in file
            self.IncrementSize()
        if opened:
            self.Close()

    def Cache(self, key: str, value, replace):
        self.cache.Put(key, value, replace)
    #endregion

    #region Get
    def GetValue(self, key: str, cacheResult=True):
        value = self.TryGetFromCache(key)
        if value is None:
            value = self.TryGetFromFile(key)
            if value is not None and cacheResult:
                self.Cache(key, value, False) # No need to cache this if it was loaded from cache anyway
        return value

    def TryGetFromCache(self, key: str):
        try:
            return self.cache.Get(key)
        except KeyError:
            return None

    def TryGetFromFile(self, key: str):
        opened = self.Open()
        try:
            return self.d[key]
        except KeyError:
            return None
        finally:
            if opened:
                self.Close()
    #endregion

    def Size(self):
        return self.GetValue(Dataset.KEY_SIZE, cacheResult=False)

    def WriteSize(self):
        opened = self.Open()
        self.d[Dataset.KEY_SIZE] = self.size # Write directly to file (do not use internal 'Put()' method)
        if opened:
            self.Close()

    #region Delete
    def Delete(self, key: str):
        self.DeleteFromFile(key)
        self.DeleteFromCache(key)

    def DeleteFromFile(self, key: str):
        opened = self.Open()
        exists = key in self.d
        if exists:
            del self.d[key]
            self.DecrementSize()
        if opened:
            self.Close()
        return exists

    def DeleteFromCache(self, key: str):
        self.cache.Delete(key)
    #endregion

    def Open(self, writeback=False):
        if not self.isOpen:
            self.d = shelve.open(self.filename, writeback=writeback)
            self.isOpen = True
            return True
        return False

    def Close(self):
        if self.isOpen:
            self.d.close()
            self.isOpen = False
            return True
        return False

    def HasNextLoad(self) -> bool:
        return self.loadStart < self.Size()

    @abstractmethod
    def NextLoad(self, loadSize: int, shuffle: bool, cacheResult: bool):
        pass