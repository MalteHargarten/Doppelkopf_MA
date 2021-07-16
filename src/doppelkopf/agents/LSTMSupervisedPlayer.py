import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
import numpy as np
from typing import List
from doppelkopf.game.Card import Card
from doppelkopf.game.Client import Client
from doppelkopf.agents.Agent import Agent
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.models.LSTMModel import LSTMModel
from doppelkopf.game.CardFeedback import CardFeedback
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class LSTMSupervisedPlayer(Agent):
    CARD_WAS_NOT_OK_VALUE = -100.0 # Set cards that were not okay to a high, negative number

    def __init__(self, loadWeightsPath: str):
        super(LSTMSupervisedPlayer, self).__init__() # Call base constructor
        self.lstm = LSTMModel.Create("Supervised LSTM", tf.float32, GameState.SIZE_STATE, [], Card.NUM_CARDTYPES, 0.001) # Learning rate does not matter here because we do not train in the Player
        if not self.lstm.TryLoadWeights(loadWeightsPath):
            raise ValueError("Could not load LSTM weights from file")
        self.lastPredictions = None
        self.cardWasNotOkayCounters = {}
        self.gameStates = []
        self.resetCardCounters()
        
    def resetCardCounters(self):
        for i in range(Doppelkopf.MAX_CARDS_PER_PLAYER):
            self.cardWasNotOkayCounters[i] = 0

    def instantiateClient(self) -> Client: # is called in base class constructor
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onCardWasOk=self.onCardWasOk,
            onCardWasNotOk=self.onCardWasNotOkay,
            onStateReceived=self.receiveGameState,
            onGameCompleted=self.onGameCompleted
        ) # Subscribe to events

    def receiveGameState(self, state: GameState, isFinalState: bool):
        self.gameStates.append(state.Flat())

    def onGameCompleted(self, winnerTeamName, score):
        self.lstm.ResetLayerStates()

    def onCardWasOk(self, state, card, feedback):
        self.lastPredictions = None # Delete the last prediction

    def onCardWasNotOkay(self, state: GameState, card: Card, feedback: CardFeedback, trickIndex: int):
        self.cardWasNotOkayCounters[trickIndex] += 1 # Increase counter
        self.lastPredictions[card.cardType] = LSTMSupervisedPlayer.CARD_WAS_NOT_OK_VALUE # Set this cardType's estimate to a negative value because it is not allowed

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        if self.lastPredictions is None: # If we do not yet have a prediction for the current situation, make one
            states = np.expand_dims(np.array(self.gameStates), axis=0) # Make a numpy array out of the gamestates list (the states are flattened already)
            self.lastPredictions = self.lstm.QValuesNumpy(inputs=states, useLastState=True)[0,-1].reshape(-1) # Pick the last timestep and make it 1D since we are dealing with a single timestep anyway
            self.gameStates.clear() # Clear the list of game states (so as to not run them again in the next 'pickCard()' call)
        # If we do not have a prediction yet, we also have not tried any
        #   cards in this state yet, so 'wrongCardTypes' is always empty
        #   (meaning we do not have to check whether the card we pick has been tried before)
        # If we have a prediction from before, then all wrongCardTypes have already
        #   been set to a negative value in the 'onCardWasNotOk' callback
        #   (meaning, once again, that we do not have to check whether the card we pick has been tried before)
        cardType = int(np.argmax(self.lastPredictions)) # Pick the highest estimated Q-value
        card = self.client.myPlayer.tryGetCardFromHand(cardType)
        return card # Try playing the picked card

    def PlayGames(self, numOfGames):
        self.resetCardCounters()
        success = super(LSTMSupervisedPlayer, self).PlayGames(numOfGames) # Call base method
        for trickIndex, counter in self.cardWasNotOkayCounters.items():
            Console.WriteInfo("%d wrong cards occured in trick %d" % (counter, trickIndex), self.name)
        return success