import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.game.Player import Player
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.agents.DNNPlayer import DNNPlayer
from doppelkopf.game.CardFeedback import CardFeedback

class CardRecord():
    def __init__(self, player: Player, state: GameState, cards: List[Card], confidenceValues: List[float], feedbacks: List[CardFeedback], predictions: np.ndarray, legalMask: np.ndarray):
        self.player = player
        self.state = state # The state in which a card was requested from the Agent
        self.cards = cards.copy() # All the cards that were picked (and possibly rejeceted) in this state
        self.confidenceValues = confidenceValues.copy() # The original prediction values for all the picked cards
        self.feedbacks = feedbacks.copy() # The feedbacks for the picked cards
        self.predictions = np.copy(predictions) # The original predictions (before they were altered by wrong cards)
        self.legalMask = np.copy(legalMask) # The true legal mask for this state

    def __str__(self) -> str:
        r = "The cards on the table are: " if len(self.state.currentStack) > 0 else "Player %s is starting the trick!" % (self.player.name)
        for card in self.state.currentStack:
            r += "\n%s" % (card)
        for card, confidence, feedback in zip(self.cards, self.confidenceValues, self.feedbacks):
            r += "\nPlayer %s tried playing %s with a confidence of %f. Server feedback was: %s" % (self.player.name, card, confidence, feedback)
        r += "\nThey could have played: "
        for i in range(len(self.legalMask)):
            if self.legalMask[i] > 0:
                r += "\n%s (prediction: %f)" % (Card.CARDTYPES[i], self.predictions[i])
        return r

    def __repr__(self) -> str:
        return self.__str__()

class TrickRecord():
    def __init__(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int, cardRecord: CardRecord, finalState: GameState):
        self.isTrickWinner = isTrickWinner
        self.isTeamMateTrickWinner = isTeamMateTrickWinner
        self.trickValue = trickValue
        self.cardRecord = cardRecord # A record of all cards that were attempted in this trick
        self.finalState = finalState # The trick's final state

    def __str__(self) -> str:
        r = "Player %s %s this trick" % (self.cardRecord.player.name, "won" if self.isTrickWinner else "didn't win")
        if not self.isTrickWinner and self.isTeamMateTrickWinner: # If we didn't win, but our teammate did
            r += ", but our team mate seems to have won it!"
        r += "\nThe trick is worth %d points" % (self.trickValue)
        r += "\n%s" % (self.cardRecord)
        return r

    def __repr__(self) -> str:
        return self.__str__()

class GameRecord():
    def __init__(self, trickRecords: List[TrickRecord]):
        self.trickRecords = trickRecords

    def __str__(self) -> str:
        r = ""
        for i, trickRecord in enumerate(self.trickRecords):
            r += "\nTrick %d: %s" % (i, trickRecord)
        return r

    def __repr__(self) -> str:
        return self.__str__()

class DNNPlayerRecorder(DNNPlayer):
    def __init__(self, learningRate: float, loadWeightsPath: str, denseLayerUnits: List[int], recordFile: str):
        super(DNNPlayerRecorder, self).__init__(learningRate, loadWeightsPath, denseLayerUnits)
        self.recordFile = recordFile
        self.trickRecords: List[TrickRecord] = []
        self.finalState = None
        self.pickCardState = None
        self.legalMask = None
        self.pickedCards = []
        self.confidenceValues = []
        self.feedbacks = []
        self.predictionCopy = None

    def instantiateClient(self) -> Client: # is called in base class constructor
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onGameStarted=self.onGameStarted,
            onCardWasOk=self.onCardWasOk,
            onCardWasNotOk=self.onCardWasNotOkay,
            onStateReceived=self.receiveGameState,
            onTrickCompleted=self.onTrickCompleted,
            onGameCompleted=self.onGameCompleted
        ) # Subscribe to events

    def onGameStarted(self):
        self.trickRecords.clear() # Clear list of records
        self.finalState = None
        self.pickCardState = None
        self.legalMask = None
        self.pickedCards.clear()
        self.confidenceValues.clear()
        self.feedbacks.clear()
        self.predictionCopy = None

    def onTrickCompleted(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int):
        super(DNNPlayerRecorder, self).onTrickCompleted(isTrickWinner, isTeamMateTrickWinner, trickValue) # Call base method
        cardRecord = CardRecord(self.client.myPlayer, self.pickCardState, self.pickedCards, self.confidenceValues, self.feedbacks, self.predictionCopy, self.legalMask)
        self.trickRecords.append(TrickRecord(isTrickWinner, isTeamMateTrickWinner, trickValue, cardRecord, self.finalState))
        self.pickCardState = None
        self.pickedCards.clear()
        self.confidenceValues.clear()
        self.feedbacks.clear()
        self.predictionCopy = None
        self.finalState = None
        self.legalMask = None

    def onGameCompleted(self, isGameWinner: bool, score: int):
        super(DNNPlayerRecorder, self).onGameCompleted(isGameWinner, score) # Call base method
        File.Append(GameRecord(self.trickRecords), self.recordFile)

    def onCardWasOk(self, state, card, feedback):
        super(DNNPlayerRecorder, self).onCardWasOk(state, card, feedback)
        self.feedbacks.append(feedback)

    def onCardWasNotOkay(self, state: GameState, card: Card, feedback: CardFeedback, trickIndex: int):
        super(DNNPlayerRecorder, self).onCardWasNotOkay(state, card, feedback, trickIndex)
        self.feedbacks.append(feedback)

    def receiveGameState(self, state: GameState, isFinalState: bool):
        if isFinalState:
            self.finalState = state

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        if self.lastPredictions is None: # If we do not have a prediction for this state yet, make one
            x = np.expand_dims(state.Flat(), axis=0) # Make the state 2D by adding the 'batch' dimension
            self.lastPredictions = self.model.QValuesNumpy(x).reshape(-1) # Contains 24 ones or zeros
            self.predictionCopy = np.copy(self.lastPredictions) # Make a copy (shallow copy, but since array contains floats, it's equivalent to a deep copy)
            self.pickCardState = state
        cardType = int(np.argmax(self.lastPredictions)) # pick the one with the highest estimated 'legal' value
        card = self.client.myPlayer.tryGetCardFromHand(cardType)
        if self.legalMask is None:
            self.legalMask = Doppelkopf.getLegalPlayableMask(state.currentStack[0] if len(state.currentStack) > 0 else None, self.client.myPlayer)
        self.pickedCards.append(card) # Store this card
        self.confidenceValues.append(self.lastPredictions[card.cardType]) # Store the confidence value for this card
        return card