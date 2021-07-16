from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.agents.Agent import Agent
from doppelkopf.utils.Helper import Helper
from doppelkopf.game.GameState import GameState
from doppelkopf.reports.PlayReport import PlayReport

class RulebasedPlayer(Agent):
    def __init__(self):
        super(RulebasedPlayer, self).__init__() # Call base constructor

    def instantiateClient(self) -> Client:
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onTrickCompleted=self.onTrickCompleted,
            onGameCompleted=self.onGameCompleted
        )
    
    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        return self.client.myPlayer.tryFollowSuit(state.currentStack[0] if len(state.currentStack) > 0 else None) # To keep this game going, pick only valid cards

    def LogReport(self, logFile, numOfGames):
        report = PlayReport(Helper.DateTimeNowToString(), numOfGames, self.client.cardsPickedCounter, self.client.okCardCounter, self.client.notInHandCounter, self.client.notAllowedCounter, self.client.gamesCompletedCounter, self.client.gamesWonCounter, self.client.gamesLostCounter, self.trickScores, self.gameScores)
        File.Append(report, logFile) # Write result object to file