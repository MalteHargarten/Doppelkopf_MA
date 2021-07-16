import numpy as np
from typing import List
from doppelkopf.models.DNN import DNN
from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.agents.Agent import Agent
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.reports.DNNPlayReport import DNNPlayReport
from doppelkopf.game.CardFeedback import CardFeedback

class DNNPlayer(Agent):
    CARD_WAS_NOT_OK_VALUE = -100.0 # Set cards, that were not okay, to a high, negative number

    def __init__(self, learningRate: float, loadWeightsPath: str, denseLayerUnits: List[int]):
        super(DNNPlayer, self).__init__() # Call base constructor
        self.loadWeightsPath = loadWeightsPath
        self.learningRate = learningRate
        self.denseLayerUnits = denseLayerUnits
        self.model = DNN("DNNPlayer", learningRate, denseLayerUnits)
        if not self.model.TryLoadWeights(loadWeightsPath):
            Console.WriteWarning("Could not load weights from file! Player will likely not perform very well")
        self.lastPredictions = None
        self.totalCardWasNotOkayCounter = 0
        self.cardWasNotOkayCounters = {}
        self.resetCardCounters()

    def instantiateClient(self) -> Client: # is called in base class constructor
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onCardWasOk=self.onCardWasOk,
            onCardWasNotOk=self.onCardWasNotOkay,
            onTrickCompleted=self.onTrickCompleted,
            onGameCompleted=self.onGameCompleted
        ) # Subscribe to events

    def onCardWasOk(self, state, card, feedback):
        self.lastPredictions = None # Delete the last prediction

    def onCardWasNotOkay(self, state: GameState, card: Card, feedback: CardFeedback, trickIndex: int):
        self.totalCardWasNotOkayCounter += 1
        self.cardWasNotOkayCounters[trickIndex] += 1 # Increase counter
        #Console.WriteInfo("Picked a wrong card (%s). Prediction for this card was %f" % (card, self.lastPredictions[card.cardType]), self.name)
        self.lastPredictions[card.cardType] = DNNPlayer.CARD_WAS_NOT_OK_VALUE # Set this cardType's estimate to a negative value because it is not allowed

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        if self.lastPredictions is None: # If we do not have a prediction for this state yet, make one
            x = np.expand_dims(state.Flat(), axis=0) # Make the state 2D by adding the 'batch' dimension
            self.lastPredictions = self.model.QValuesNumpy(x).reshape(-1) # Contains 24 ones or zeros
        # If we do not have a prediction yet, we also have not tried any
        #   cards in this state yet, so 'wrongCardTypes' is always empty
        #   (meaning we do not have to check whether the card we pick has been tried before)
        # If we have a prediction from before, then all wrongCardTypes have already
        #   been set to a negative value in the 'onCardWasNotOk' callback
        #   (meaning, once again, that we do not have to check whether the card we pick has been tried before)
        cardType = int(np.argmax(self.lastPredictions)) # pick the one with the highest estimated 'legal' value
        card = self.client.myPlayer.tryGetCardFromHand(cardType)
        return card # Try playing the picked card type

    def PlayGames(self, numOfGames):
        self.resetCardCounters()
        success = super(DNNPlayer, self).PlayGames(numOfGames) # Call base method
        if self.totalCardWasNotOkayCounter > 0: # Only if there were any wrong cards to begin with...
            self.printCardCounters() # ... print the whole dictionary
        return success
            
    def resetCardCounters(self):
        self.totalCardWasNotOkayCounter = 0
        for i in range(Doppelkopf.MAX_CARDS_PER_PLAYER):
            self.cardWasNotOkayCounters[i] = 0

    def printCardCounters(self):
        for trickIndex, counter in self.cardWasNotOkayCounters.items():
            Console.WriteInfo("%d wrong cards occured in trick %d" % (counter, trickIndex), self.name)

    def LogReport(self, logFile, numOfGames):
        report = DNNPlayReport(Helper.DateTimeNowToString(), self.loadWeightsPath, numOfGames, self.client.cardsPickedCounter, self.client.okCardCounter, self.client.notInHandCounter, self.client.notAllowedCounter, self.client.gamesCompletedCounter, self.client.gamesWonCounter, self.client.gamesLostCounter, self.trickScores, self.gameScores, self.cardWasNotOkayCounters)
        File.Append(report, logFile) # Write result object to file