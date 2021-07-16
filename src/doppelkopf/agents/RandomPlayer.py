import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.agents.Agent import Agent
from doppelkopf.utils.Helper import Helper
from doppelkopf.game.GameState import GameState
from doppelkopf.reports.PlayReport import PlayReport

class RandomPlayer(Agent):
    def __init__(self):
        super(RandomPlayer, self).__init__()

    def instantiateClient(self) -> Client:
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onTrickCompleted=self.onTrickCompleted,
            onGameCompleted=self.onGameCompleted
        )

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        cardTypes = [i for i in range(Card.NUM_CARDTYPES) if i not in wrongCardTypes] # List of all cardTypes that haven't been tried yet
        cardType = cardTypes[int(np.random.uniform(high=len(cardTypes)))] # Pick any card type from the list
        return self.client.myPlayer.tryGetCardFromHand(cardType)

    def LogReport(self, logFile, numOfGames):
        report = PlayReport(Helper.DateTimeNowToString(), numOfGames, self.client.cardsPickedCounter, self.client.okCardCounter, self.client.notInHandCounter, self.client.notAllowedCounter, self.client.gamesCompletedCounter, self.client.gamesWonCounter, self.client.gamesLostCounter, self.trickScores, self.gameScores)
        File.Append(report, logFile) # Write result object to file