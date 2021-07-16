import threading
from typing import List
from abc import ABC, abstractmethod
from doppelkopf.utils.Console import Console
from doppelkopf.game.Card import Card
from doppelkopf.game.Client import Client
from doppelkopf.game.Player import Player
from doppelkopf.game.GameState import GameState

# Make class abstract by inheriting from ABC (abc module)
class Agent(ABC):
    def __init__(self):
        self.name = None
        self.client = self.instantiateClient()
        self.trickScores = []
        self.gameScores = []

    def receivePlayer(self, player: Player):
        self.name = "Agent %s" % player.name

    def onTrickCompleted(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int):
        if isTrickWinner or isTeamMateTrickWinner:
            self.trickScores.append(trickValue)
        else:
            self.trickScores.append(trickValue * (-1)) # Add the negative trick value to the list

    def onGameCompleted(self, isGameWinner: bool, score: int):
        self.gameScores.append(score)

    def ConnectToServer(self, host, port):
        self.client.Connect(host, port)

    def DisconnectFromServer(self):
        self.client.Disconnect()

    def threadInterruptPlaying(self):
        Console.ReadInput("Press Enter to stop/end gameplay\n") # Wait for user input
        self.client.Stop() # Stop playing games (if it hasn't stopped already)

    def PlayGames(self, numOfGames=None, canBeInterrupted=True):
        if canBeInterrupted:
            thread = threading.Thread(target=self.threadInterruptPlaying, daemon=True) # The python program quits when only daemons are left alive (https://docs.python.org/3/library/threading.html)
            thread.start() # Start input reading thread
        return self.client.PlayGames(numOfGames)

    @abstractmethod
    def instantiateClient(self) -> Client:
        pass

    @abstractmethod
    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        pass