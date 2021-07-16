import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
import numpy as np
import tensorflow as tf
from typing import List
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from doppelkopf.game.Card import Card
from doppelkopf.models.DNN import DNN
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.agents.Agent import Agent
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.models.LSTMModel import LSTMModel
from doppelkopf.reports.LSTMPlayReport import LSTMPlayReport
from doppelkopf.game.CardFeedback import CardFeedback

class LSTMPlayer(Agent):
    CARD_WAS_NOT_OK_VALUE = -100.0 # Set cards that were not okay to a high, negative number

    def __init__(self, dnnWeightsPath: str, lstmWeightsPath: str, denseLayerUnits:List[int], hiddenLSTMUnits: List[int]):
        super(LSTMPlayer, self).__init__() # Call base constructor
        self.dnnWeightsPath = dnnWeightsPath
        self.lstmWeightsPath = lstmWeightsPath
        self.dnn = DNN("DNN_Model", 0.001, denseLayerUnits=denseLayerUnits) # Learning rate is irrelevant, because the DNN is not trained within the LSTMPlayer
        if not self.dnn.TryLoadWeights(dnnWeightsPath): # Load DNN from file (this agent requires that the DNN has already been trained)
            raise ValueError("Could not load DNN weights from file")
        self.lstm = LSTMModel.Create(
            name="LSTM Model",
            inputType=tf.float32,
            inputSize=GameState.SIZE_STATE + Card.NUM_CARDTYPES, # 416 + 24 (append DNN's output to state)
            hiddenLSTMUnits=hiddenLSTMUnits, # Old layers: [256, 128, 64, 32]
            outputUnits=Card.NUM_CARDTYPES,
            learningRate=0.001) # Learning rate is irrelevant, because the LSTM is not trained within the LSTMPlayer
        if lstmWeightsPath is None or not self.lstm.TryLoadWeights(lstmWeightsPath): # Load LSTM from file (this agent requires that the DNN has already been trained)
            Console.WriteWarning("Could not load LSTM weights from file! Player will likely not perform very well")
        self.cardWasNotOkayCounters = {}
        self.resetCardCounters()
        self.gameStates = []
        self.lastPredictions = None

    def resetCardCounters(self):
        for i in range(Doppelkopf.MAX_CARDS_PER_PLAYER):
            self.cardWasNotOkayCounters[i] = 0

    def instantiateClient(self) -> Client: # is called in base class constructor
        return Client(
            onCardRequested=self.pickCard,
            onCardWasOk=self.onCardWasOk,
            onCardWasNotOk=self.onCardWasNotOkay,
            onPlayerReceived=self.receivePlayer,
            onStateReceived=self.receiveGameState,
            onGameCompleted=self.onGameCompleted
        ) # Subscribe to events

    def receiveGameState(self, state: GameState, isFinalState: bool):
        self.gameStates.append(state.Flat())

    def onGameCompleted(self, winnerTeamName, score):
        self.lstm.ResetLayerStates()

    def onCardWasOk(self, state, card, feedback):
        self.lastPredictions = None # Delete this prediction so that next time, we make a new one when asked for a card

    def onCardWasNotOkay(self, state: GameState, card: Card, feedback: CardFeedback, trickIndex):
        self.cardWasNotOkayCounters[trickIndex] += 1 # Increase counter
        self.lastPredictions[card.cardType] = LSTMPlayer.CARD_WAS_NOT_OK_VALUE # Set this cardType's estimate to a negative value because it is not allowed

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        # Do Exploitation
        if self.lastPredictions is None: # If we do not yet have a prediction for the current situation, make one
            states = np.array(self.gameStates) # Make a numpy array out of the gamestates list (the states are flattened already)
            dnnEstimates = self.dnn.predict(x=states) # Run all the game states (since the last 'pickCard()' call)
            lstmInput = np.expand_dims(np.concatenate((states, dnnEstimates), axis=1), axis=0)
            self.lastPredictions = self.lstm.QValuesNumpy(inputs=lstmInput, useLastState=True)[0,-1].reshape(-1) # Make it 1D since we are dealing with a single timestep anyway
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
        success = super(LSTMPlayer, self).PlayGames(numOfGames) # Call base method
        self.printCardCounters()
        return success

    def printCardCounters(self):
        for trickIndex, counter in self.cardWasNotOkayCounters.items():
            Console.WriteWarning("%d wrong cards occured in trick %d" % (counter, trickIndex), self.name)

    def LogReport(self, logFile, numOfGames):
        report = LSTMPlayReport(Helper.DateTimeNowToString(), self.dnnWeightsPath, self.lstmWeightsPath, numOfGames, self.client.cardsPickedCounter, self.client.okCardCounter, self.client.notInHandCounter, self.client.notAllowedCounter, self.client.gamesCompletedCounter, self.client.gamesWonCounter, self.client.gamesLostCounter, self.cardWasNotOkayCounters)
        File.Append(report, logFile) # Write result object to file