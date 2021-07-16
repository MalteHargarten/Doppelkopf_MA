import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from typing import List
from doppelkopf.models.DNN import DNN
from doppelkopf.game.Card import Card
from doppelkopf.utils.File import File
from doppelkopf.game.Client import Client
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.agents.EnumsRL import RewardMode
from doppelkopf.agents.EnumsRL import RewardType
from doppelkopf.data.Experience import Experience
from doppelkopf.game.Doppelkopf import Doppelkopf
from doppelkopf.agents.EnumsRL import PickCardMode
from doppelkopf.agents.DNNPlayer  import DNNPlayer
from doppelkopf.game.CardFeedback import CardFeedback
from doppelkopf.data.ReplayBuffer import ReplayBuffer
from doppelkopf.reports.DNNTrainReportRL import DNNTrainReportRL

class DNNTrainerRL(DNNPlayer):
    INVALID_CARD_REWARD = -1.0 # This may be subject to change

    def __init__(self, learningRate: float, loadWeightsPath: str, saveWeightsPath: str, denseLayerUnits: List[int], copyWeightsInterval: int, bufferSize: int, batchSize: int, numOfBatches: int, discountFactor, epsilon, epsilonDecayRate, minimumEpsilon, rewardType):
        super(DNNTrainerRL, self).__init__(learningRate, loadWeightsPath, denseLayerUnits) # Call base constructor (creates DNN and LSTMModel)
        # # # # # # # # # # # # # # # Create target Network for target calculation # # # # # # # # # # # # # # #
        self.saveWeightsPath = saveWeightsPath
        self.copyWeightsInterval = copyWeightsInterval
        self.bufferSize = bufferSize
        self.buffer = ReplayBuffer(cacheSize=self.bufferSize)
        self.batchSize = batchSize
        self.numOfBatches = numOfBatches # The number of batches sampled after every completed game
        if numOfBatches <= 0:
            raise ValueError("'numOfBatches' was %d, but must not be zero or less!" % numOfBatches)
        self.numOfSamples = numOfBatches * self.batchSize
        if self.numOfSamples > self.bufferSize:
            raise ValueError("You are trying to request more samples from the buffer than can fit in the buffer!")
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonDecayRate = epsilonDecayRate
        self.minimumEpsilon = minimumEpsilon
        self.pickCardMode = PickCardMode.EXPLORATION
        self.acceptedCards: List[Card] = [] # The card(s) that was/were last picked by the player/trainer
        self.pickCardStates: List[GameState] = [] # The state(s) where the player/trainer is ordered to pick a card
        self.rewardFunc = None
        self.trainingCounter = 0 # A counter for how often we call any of the Train() methods
        self.rewardType = rewardType
        self.rewardMode = None
        self.useTargetNetwork = False # Is assigned a reasonable value in the "setRewardType()" method
        self.setRewardType(rewardType) # Assigs self.useTargetNetwork as needed
        self.targetNetwork = None
        if self.useTargetNetwork:
            self.targetNetwork = DNN("Target Network", learningRate, denseLayerUnits, loadWeightsPath=loadWeightsPath)
        self.lastLoss = None

    def instantiateClient(self) -> Client:
        return Client(
            onCardRequested=self.pickCard,
            onPlayerReceived=self.receivePlayer,
            onCardWasOk=self.onCardWasOk,
            onCardWasNotOk=self.onCardWasNotOkay,
            onTrickCompleted=self.onTrickCompleted,
            onGameCompleted=self.onGameCompleted,
        )

    def onTrickCompleted(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int):
        super(DNNTrainerRL, self).onTrickCompleted(isTrickWinner, isTeamMateTrickWinner, trickValue)
        if self.rewardMode == RewardMode.PER_TRICK:
            # Get the last state, in which a card was requested from the agent (there should be only one anyway)
            state = self.pickCardStates[-1]
            # Get the last card that was accepted (there should be only one anyway)
            action = self.acceptedCards[-1].cardType
            reward = self.rewardFunc(isTrickWinner=isTrickWinner, isTeamMateTrickWinner=isTeamMateTrickWinner, trickValue=trickValue)
            #Console.WriteInfo("Reward was %f" % reward, self.name)
            newExperience = Experience(state.Flat(), action, reward, None) # In this reward mode, next states don't matter
            self.buffer.PutNext(newExperience)
            Console.WriteDebug("Added new experience to buffer: %s" % (newExperience), self.name + " onTrickCompleted()")
            self.pickCardStates.clear()
            self.acceptedCards.clear()

    def onGameCompleted(self, isGameWinner: bool, score):
        super(DNNTrainerRL, self).onGameCompleted(isGameWinner, score)
        if self.rewardMode == RewardMode.PER_GAME:
            reward = self.rewardFunc(isGameWinner=isGameWinner, score=score)
            for i, (state, pickedCard) in enumerate(zip(self.pickCardStates, self.acceptedCards)):
                nextState = self.pickCardStates[i + 1].Flat() if (i + 1) < len(self.pickCardStates) else None
                action = pickedCard.cardType
                newExperience = Experience(state.Flat(), action, reward, nextState)
                self.buffer.PutNext(newExperience)
                Console.WriteDebug("Added new experience to buffer: %s" % (newExperience), self.name + " onGameCompleted()")
            self.pickCardStates.clear()
            self.acceptedCards.clear()
        self.applyEpsilonDecay() # ToDo: Decide whether to apply epsilon decay depending on win/lose scenario (or reward based)
        if self.buffer.IsFull(): # Only once the buffer is full, start training
            self.Train() # Train a handful of batches
        else:
            Console.WriteInfo("Cannot train yet, buffer only has %d experiences" % (self.buffer.Size()), self.name)
    
    def onCardWasOk(self, state: GameState, card: Card, feedback: CardFeedback):
        self.acceptedCards.append(card) # Only accepted cards are added to the list
        super(DNNTrainerRL, self).onCardWasOk(state, card, feedback) # Delete the last prediction
        if self.rewardMode == RewardMode.PER_CARD:
            reward = self.rewardFunc(feedback) # Get reward for this (valid) card
            action = card.cardType
            newExperience = Experience(state.Flat(), action, reward, None) # In this reward mode, next states don't matter
            self.buffer.PutNext(newExperience)
            Console.WriteDebug("Added new experience to buffer: %s" % (newExperience), self.name + " onCardWasOk()")

    def onCardWasNotOkay(self, state: GameState, card: Card, feedback: CardFeedback, trickIndex: int):
        self.totalCardWasNotOkayCounter += 1
        self.cardWasNotOkayCounters[trickIndex][self.pickCardMode] += 1 # Increase counter
        if self.lastPredictions is not None:
            self.lastPredictions[card.cardType] = DNNTrainerRL.CARD_WAS_NOT_OK_VALUE # Set this cardType's estimate to a negative value because it is not allowed
            #Console.WriteInfo("Set the last Predictions of this card to a negative value to make sure it's not picked again", self.name)
        # # # # # # # # # # # # # # # Do reward for invalid card # # # # # # # # # # # # # # #
        # This is performed regardless of the current rewardMode.
        # In RewardMode.PER_CARD, this action is treated just like any other selected card
        # In RewardMode.PER_TRICK, all previous tricks have already been processed, but this trick cannot be processed, 
        #   as it is incomplete until all players have picked a valid card. So we just process the invalid card here and 
        #   wait for a valid one to ba played
        # In RewardMode.PER_GAME, the game is not complete, so it cannot be processed yet. However, this card can.
        reward = DNNTrainerRL.INVALID_CARD_REWARD
        newExperience = Experience(state.Flat(), card.cardType, reward, None) # In either reward mode, invalid cards have no next state
        self.buffer.PutNext(newExperience)
        Console.WriteDebug("Added new experience to buffer: %s" % (newExperience), self.name + " onCardWasNotOkay()")

    def pickCardExploration(self, wrongCardTypes: List[int]) -> Card:
        cardTypes = [i for i in range(Card.NUM_CARDTYPES) if i not in wrongCardTypes] # List of all cardTypes that haven't been tried yet
        cardType = cardTypes[int(np.random.uniform(high=len(cardTypes)))] # Pick any card type from the list
        return self.client.myPlayer.tryGetCardFromHand(cardType)

    def pickCardExploitation(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        # Do Exploitation
        if self.lastPredictions is None: # If we do not yet have a prediction for the current situation, make one
            x = np.expand_dims(state.Flat(), axis=0) # Make the state 2D by adding the 'batch' dimension
            self.lastPredictions = self.model.QValuesNumpy(x).reshape(-1) # Make it 1D since we are dealing with a single state anyway
            # If we do not have a prediction yet, we have to make sure that any cards that
            #   may have been tried before (by exploration) are marked as invalid
            for wrongCardType in wrongCardTypes:
                self.lastPredictions[wrongCardType] = DNNTrainerRL.CARD_WAS_NOT_OK_VALUE
            #Console.WriteInfo("Made predictions for this state as there weren't any before", self.name)
        # If we have a prediction from before, then all wrongCardTypes have already
        #   been set to a negative value in the 'onCardWasNotOk' callback
        #   (meaning that we do not have to check whether the card we pick has been tried before)
        cardType = int(np.argmax(self.lastPredictions)) # Pick the highest estimated Q-value
        card = self.client.myPlayer.tryGetCardFromHand(cardType)
        return card # Try playing the picked card

    def pickCard(self, state: GameState, wrongCardTypes: List[int]) -> Card:
        if state not in self.pickCardStates: # If this state is not in the list yet
            self.pickCardStates.append(state) # Get the state and add it to the list of states
        chosenCard = None
        if np.random.uniform() > self.epsilon:
            # Do exploitation
            Console.WriteDebug("Doing exploitation!", self.name)
            self.pickCardMode = PickCardMode.EXPLOITATION
            chosenCard = self.pickCardExploitation(state, wrongCardTypes) # Make prediction, if non exists already
        else:
            # Do exploration
            Console.WriteDebug("Doing exploration!", self.name)
            self.pickCardMode = PickCardMode.EXPLORATION
            chosenCard = self.pickCardExploration(wrongCardTypes)
        return chosenCard # Try playing the picked card type
    #endregion

    #region Reward functions
    def setRewardType(self, rewardType: RewardType):
        if rewardType == RewardType.PER_VALID_CARD:
            self.rewardFunc = self.rewardPerValidCard
            self.rewardMode = RewardMode.PER_CARD
            self.useTargetNetwork = False # This is a short-term goal. Long term goals have no impact on the decision making here
        elif rewardType == RewardType.PER_TRICK_SIMPLE:
            self.rewardFunc = self.rewardPerTrickSimple
            self.rewardMode = RewardMode.PER_TRICK
            self.useTargetNetwork = False # This is a short-term goal. Long term goals have no impact on the decision making here
        elif rewardType == RewardType.PER_TRICK_PROPORTIONAL:
            self.rewardFunc = self.rewardPerTrickProportional
            self.rewardMode = RewardMode.PER_TRICK
            self.useTargetNetwork = False # This is a short-term goal. Long term goals have no impact on the decision making here
        elif rewardType == RewardType.PER_TRICK_PROPORTIONAL_FIXED_RATE:
            self.rewardFunc = self.rewardPerTrickProportionalFixedRate
            self.rewardMode = RewardMode.PER_TRICK
            self.useTargetNetwork = False # This is a short-term goal. Long term goals have no impact on the decision making here
        elif rewardType == RewardType.PER_GAME_SIMPLE:
            self.rewardFunc = self.rewardPerGameSimple
            self.rewardMode = RewardMode.PER_GAME
            self.useTargetNetwork = True # This is a long-term goal, that requires future rewards to play a role as well as short-term rewards
        elif rewardType == RewardType.PER_GAME_PROPORTIONAL:
            self.rewardFunc = self.rewardPerGameProportional
            self.rewardMode = RewardMode.PER_GAME
            self.useTargetNetwork = True # This is a long-term goal, that requires future rewards to play a role as well as short-term rewards
        elif rewardType == RewardType.PER_GAME_PROPORTIONAL_FIXED_RATE:
            self.rewardFunc = self.rewardPerGameProportionalFixedRate
            self.rewardMode = RewardMode.PER_GAME
            self.useTargetNetwork = True # This is a long-term goal, that requires future rewards to play a role as well as short-term rewards
        Console.WriteInfo("Changed rewardType to %s" % (rewardType), self.name)

    def rewardPerValidCard(self, feedback: CardFeedback) -> float:
        if feedback == CardFeedback.NOT_ALLOWED or feedback == CardFeedback.NOT_IN_HAND:
            return -1.0
        return 1.0

    def rewardPerTrickSimple(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue=None) -> float:
        return 1.0 if isTrickWinner or isTeamMateTrickWinner else -1.0

    def rewardPerTrickProportional(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int) -> float:
        reward = trickValue / Doppelkopf.MAX_POINTS_PER_TRICK # Divide value of this trick by maximum possible points (range 0-1)
        if not isTrickWinner and not isTeamMateTrickWinner:
            reward *= -1
        return reward

    def rewardPerTrickProportionalFixedRate(self, isTrickWinner: bool, isTeamMateTrickWinner: bool, trickValue: int) -> float:
        proportionalReward = self.rewardPerTrickProportional(isTrickWinner, isTeamMateTrickWinner, trickValue) # between -1 and 1
        fixedRate = 0.5 if proportionalReward >= 0 else -0.5
        return fixedRate + (proportionalReward / 2)

    def rewardPerGameSimple(self, isGameWinner: bool, score=None) -> float:
        return 1.0 if isGameWinner else -1.0

    def rewardPerGameProportional(self, isGameWinner: bool, score) -> float:
        reward = score / Doppelkopf.TOTAL_POINTS
        if not isGameWinner:
            reward *= -1
        return reward

    def rewardPerGameProportionalFixedRate(self, isGameWinner: bool, score) -> float:
        proportionalReward = self.rewardPerGameProportional(isGameWinner, score)
        fixedRate = 0.5 if proportionalReward >= 0 else -0.5
        return fixedRate + (proportionalReward / 2)
    #endregion

    #region Training related methods
    def applyEpsilonDecay(self):
        if self.epsilon > self.minimumEpsilon:
            self.epsilon = max(self.epsilon * self.epsilonDecayRate, self.minimumEpsilon) # If epsilon after decay is smaller than minimum, choose minimum

    def CalculateYValues(self, experiences: List[Experience], nextStateMask: List[int], modelEstimatesStates: np.ndarray, modelEstimatesNextStates: np.ndarray, targetNetworkEstimates: np.ndarray):
        yActuals = []
        targetMask = np.zeros(shape=modelEstimatesStates.shape).astype('bool')
        for i, experience in enumerate(experiences):
            target = experience.reward
            targetMask[i, experience.action] = True
            if self.useTargetNetwork:
                nextStateIndex = nextStateMask[i]
                Console.WriteDebug("'nextStateIndex' for experience %d is %s" % (i, str(nextStateIndex)), self.name)
                if nextStateIndex is not None: # If we use a target network and there is a 'next state'
                    qNetworkArgmax = np.argmax(modelEstimatesNextStates[nextStateIndex]) # Index of highest estimated Q-value for the next state
                    Console.WriteDebug("'qNetworkArgmax' for experience %d is %s" % (i, str(qNetworkArgmax)), self.name)
                    targetNetworkQValue = targetNetworkEstimates[nextStateIndex, qNetworkArgmax] # Q-value for next state, as estimated by Target network
                    Console.WriteDebug("'targetNetworkQValue' for experience %d is %s" % (i, str(targetNetworkQValue)), self.name)
                    target += self.discountFactor * targetNetworkQValue
            yActuals.append(target)
        return tf.constant(np.array(yActuals, dtype='float32')), tf.constant(targetMask)

    def Train(self):
        # Get random samples from the training set
        Console.WriteDebug("Sampling %d samples from buffer" % (self.numOfSamples), self.name)
        experiences = self.buffer.RandomBatch(self.numOfSamples) # Get a number of samples and use them for training
        self.lastLoss = 0
        for i in range(self.numOfBatches):
            start = i * self.batchSize
            end = (i + 1) * self.batchSize
            self.lastLoss += self.TrainExperiences(experiences[start:end])
        self.lastLoss /= self.numOfBatches
        self.trainingCounter += 1
        if self.trainingCounter % self.copyWeightsInterval == (self.copyWeightsInterval - 1):
            if self.useTargetNetwork:
                self.targetNetwork.CopyWeightsFrom(self.model) # Copy the weights from the 'regular LSTMModel' to the target network
            self.model.TrySaveWeights(self.saveWeightsPath) # Store the weights every now and then so as to not lose progress if the program crashes
        Console.WriteSuccess("Batch-Training complete. Loss: %s" % (self.lastLoss), self.name)

    def TrainExperiences(self, experiences: List[Experience]) -> np.ndarray:
        states = [experience.state for experience in experiences]
        states: np.ndarray = np.array(states) # Join all the states from all the experiences together
        Console.WriteDebug("Constructed 'states' np-array of shape %s" % (str(states.shape)), self.name)
        nextStates = None
        nextStateMask = None
        if self.useTargetNetwork:
            nextStates = []
            nextStateMask = []
            for experience in experiences:
                if experience.nextState is not None:
                    nextStates.append(experience.nextState) # Only take the non-None 'next states'
                    nextStateMask.append(len(nextStates) - 1) # Add the 'index' of the next state
                else:
                    nextStateMask.append(None) # Add 'None' to indicate that this experience has no 'next state'
            nextStates = np.array(nextStates)
            Console.WriteDebug("Constructed 'nextStates' np-array of shape %s along with a mask of length %d" % (str(nextStates.shape), len(nextStateMask)), self.name)
        loss = None
        with tf.GradientTape() as tape:
            modelEstimatesStates = self.model.QValues(states)
            Console.WriteDebug("Got the Q-estimates for 'states': %s" % (modelEstimatesStates), self.name)
            modelEstimatesNextStates = None
            targetNetworkEstimates = None
            if self.useTargetNetwork and len(nextStates) > 0: # Using the target network only makes sense if there are any 'next states' to work with
                modelEstimatesNextStates = tf.keras.backend.eval(self.model.QValues(nextStates)) # Let the Q-network evaluate all the 'next states'
                Console.WriteDebug("Got the Q-estimates for 'nextStates': %s" % (modelEstimatesNextStates), self.name)
                targetNetworkEstimates = tf.keras.backend.eval(self.targetNetwork.QValues(nextStates)) # Let the Target-network evaluate all the 'next states'
                Console.WriteDebug("Got the Target Q-estimates for 'nextStates': %s" % (targetNetworkEstimates), self.name)
            yActuals, targetMask = self.CalculateYValues(experiences, nextStateMask, tf.keras.backend.eval(modelEstimatesStates), modelEstimatesNextStates, targetNetworkEstimates) # Evaluate the Tensors to get numpy arrays
            Console.WriteDebug("Constructed the target values and got: %s" % (yActuals), self.name)
            yPredicts = modelEstimatesStates[targetMask]
            Console.WriteDebug("Selected the predictions using the target mask and got: %s" % (yPredicts), self.name)
            loss = tf.keras.losses.MeanSquaredError()(yActuals, yPredicts) # https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError            
        gradients = tape.gradient(loss, self.model.trainable_variables) # Get gradients for LSTM's trainable variables
        self.model.ApplyGradients(gradients) # Apply those gradients for the calculated loss value
        return tf.keras.backend.eval(loss)

    def DoReinforcementLearning(self, numberOfGames):
        self.buffer.ClearAll()
        # # # # # # # # # # # # # # # Play games and store their recordings in buffer, then train with said buffer at the end of each game # # # # # # # # # # # # # # # 
        self.trainingCounter = 0
        self.PlayGames(numberOfGames)
        # # # # # # # # # # # # # # # Save weights to file # # # # # # # # # # # # # # #
        self.model.TrySaveWeights(self.saveWeightsPath)
        return self.lastLoss
    #endregion

    def resetCardCounters(self):
        self.totalCardWasNotOkayCounter = 0
        for i in range(Doppelkopf.MAX_CARDS_PER_PLAYER):
            self.cardWasNotOkayCounters[i] = { 
                PickCardMode.EXPLORATION: 0,
                PickCardMode.EXPLOITATION: 0,
            }

    def printCardCounters(self):
        sumExploration = 0
        sumExploitation = 0
        for trickIndex, entry in self.cardWasNotOkayCounters.items():
            sumExploration += entry[PickCardMode.EXPLORATION]
            sumExploitation += entry[PickCardMode.EXPLOITATION]
            Console.WriteInfo("Wrong cards in trick %d: Exploration: %d, Exploitation: %d" % (trickIndex, entry[PickCardMode.EXPLORATION], entry[PickCardMode.EXPLOITATION]), self.name)
        totalSum = sumExploration + sumExploitation
        Console.WriteInfo("In total, %d incorrect cards were picked. Exploration: %d, Exploitation: %d" % (totalSum, sumExploration, sumExploitation), self.name)

    def LogReport(self, logFile, numOfGames, lastLoss: float):
        report = DNNTrainReportRL(Helper.DateTimeNowToString(), self.loadWeightsPath, self.saveWeightsPath, numOfGames, self.client.cardsPickedCounter, self.client.okCardCounter, self.client.notInHandCounter, self.client.notAllowedCounter, self.client.gamesCompletedCounter, self.client.gamesWonCounter, self.client.gamesLostCounter, self.trickScores, self.gameScores, self.cardWasNotOkayCounters, self.rewardType, self.denseLayerUnits, self.batchSize, self.numOfBatches, self.bufferSize, self.copyWeightsInterval, self.discountFactor, self.epsilon, self.epsilonDecayRate, self.minimumEpsilon, lastLoss, self.learningRate)
        File.Append(report, logFile) # Write result object to file