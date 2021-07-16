import numpy as np
from typing import List, Tuple
from doppelkopf.agents.EnumsRL import RewardType
from doppelkopf.agents.EnumsRL import PickCardMode
from doppelkopf.reports.DNNPlayReport import DNNPlayReport

class DNNTrainReportRL(DNNPlayReport):
    def __init__(self, datetime: str, loadWeightsPath: str, saveWeightsPath: str, requestedNumOfGames: int, cardsPickedCounter: int, okCardCounter: int, notInHandCounter: int, notAllowedCounter: int, gamesCompletedCounter: int, gamesWonCounter: int, gamesLostCounter: int, trickScores: List[int], gameScores: List[int], invalidCardCounters: dict, rewardType: RewardType, denseLayerUnits: List[int], batchSize: int, numOfBatches: int, bufferSize: int, copyWeightsInterval: int, discountFactor: float, lastEpsilon: float, epsilonDecayRate: float, minimumEpsilon: float, lastLoss: float, learningRate: float):
        super(DNNTrainReportRL, self).__init__(datetime, loadWeightsPath, requestedNumOfGames, cardsPickedCounter, okCardCounter, notInHandCounter, notAllowedCounter, gamesCompletedCounter, gamesWonCounter, gamesLostCounter, trickScores, gameScores, invalidCardCounters)
        self.saveWeightsPath = saveWeightsPath
        self.rewardType = rewardType
        self.denseLayerUnits = denseLayerUnits
        self.batchSize = batchSize
        self.numOfBatches = numOfBatches
        self.bufferSize = bufferSize
        self.copyWeightsInterval = copyWeightsInterval
        self.discountFactor = discountFactor
        self.lastEpsilon = lastEpsilon
        self.epsilonDecayRate = epsilonDecayRate
        self.minimumEpsilon = minimumEpsilon
        self.lastLoss = lastLoss
        self.learningRate = learningRate

    def __str__(self) -> str:
        r = "Training Session recorded on %s" % (self.datetime)
        r += "\nDNN loaded weights from '%s' and saved them to %s, a learning_rate of %f and the rewardType %s" % (self.loadWeightsPath, self.saveWeightsPath, self.learningRate, self.rewardType)
        r += "\nThe DNN has the following hidden layer sizes: %s" % (str(self.denseLayerUnits))
        r += "\nDuring each training session %d batches, each with %d samples, were used for training" % (self.numOfBatches, self.batchSize)
        r += "\nThe Buffer stored %d samples during training" % (self.bufferSize)
        r += "\nEpsilon was last at a value of %f, with a decay rate of %f and a minimum of %f" % (self.lastEpsilon, self.epsilonDecayRate, self.minimumEpsilon)
        if self.lastLoss is not None:
            r += "\nThe last loss was %f" % (self.lastLoss)
        else:
            r += "\nThere is no loss recorded. Most likely, the trainer never got to train because the buffer wasn't full yet"
        r += "\nWeights were copied every %d training sessions" % (self.copyWeightsInterval)
        r += "\nFuture rewards were discounted at a value of %f" % (self.discountFactor)
        if self.gamesCompletedCounter > 0:
            if self.requestedNumOfGames is not None:
                r += "\n%d/%d games were completed successfully!" % (self.gamesCompletedCounter, self.requestedNumOfGames)
            else:
                r += "\n%d games were completed successfully!" % (self.gamesCompletedCounter)
            r += "\n%d/%d (%f %%) games were won!" % (self.gamesWonCounter, self.gamesCompletedCounter, (self.gamesWonCounter / self.gamesCompletedCounter))
        if len(self.trickScores) > 0:
            avg = np.average(self.trickScores)
            r += "\nOn average, the agent earned %f points per trick" % (avg)
        if len(self.gameScores) > 0:
            avg = np.average(self.gameScores)
            r += "\nOn average, the agent earned %f points per game" % (avg)
        r += "\nIn total, %d cards were picked" % (self.cardsPickedCounter)
        if self.cardsPickedCounter > 0:
            percentageOk = 100 * self.okCardCounter / float(self.cardsPickedCounter)
            percentageNotInHand = 100 * self.notInHandCounter / float(self.cardsPickedCounter)
            percentageNotAllowed = 100 * self.notAllowedCounter / float(self.cardsPickedCounter)
            r += "\n%d (%f %%) Cards were OK\n%d (%f %%) Cards were NOT_IN_HAND\n%d (%f %%) Cards were NOT_ALLOWED" % (self.okCardCounter, percentageOk, self.notInHandCounter, percentageNotInHand, self.notAllowedCounter, percentageNotAllowed)
        for trickIndex, entry in self.invalidCardCounters.items():
            if isinstance(entry, dict):
                r += "\nWrong cards in trick %d: Exploration: %d, Exploitation: %d" % (trickIndex, entry[PickCardMode.EXPLORATION], entry[PickCardMode.EXPLOITATION])
            elif isinstance(entry, int):
                r += "\n%d wrong cards occured in trick %d" % (entry, trickIndex)
        return r

    def ToCSV(self, includeHeader=True):
        r = ""
        if includeHeader:
            r += "Reward Type;Load Weights Path;Save Weights Path;Learning Rate;Batch Size;Dense Layer Units;Number of batches;Replay Buffer Size;Discount Factor;Copy weights Interval;Last Epsilon;Epsilon Decay Rate;Minimum Epsilon;Last Training Loss;Total cards picked;Cards OK;Cards NOT_IN_HAND;Cards NOT_ALLOWED;Requested Games;Completed Games;Games Won;Games Lost;Trick Scores;Game Scores;Wrong Cards (Exploration) Trick 1;Wrong Cards (Exploration) Trick 2;Wrong Cards (Exploration) Trick 3;Wrong Cards (Exploration) Trick 4;Wrong Cards (Exploration) Trick 5;Wrong Cards (Exploration) Trick 6;Wrong Cards (Exploration) Trick 7;Wrong Cards (Exploration) Trick 8;Wrong Cards (Exploration) Trick 9;Wrong Cards (Exploration) Trick 10;Wrong Cards (Exploration) Trick 11;Wrong Cards (Exploration) Trick 12;Wrong Cards (Exploitation) Trick 1;Wrong Cards (Exploitation) Trick 2;Wrong Cards (Exploitation) Trick 3;Wrong Cards (Exploitation) Trick 4;Wrong Cards (Exploitation) Trick 5;Wrong Cards (Exploitation) Trick 6;Wrong Cards (Exploitation) Trick 7;Wrong Cards (Exploitation) Trick 8;Wrong Cards (Exploitation) Trick 9;Wrong Cards (Exploitation) Trick 10;Wrong Cards (Exploitation) Trick 11;Wrong Cards (Exploitation) Trick 12\n"
        r += "%s;%s;%s;%f;%d;%s;%d;%d;%f;%d;%f;%f;%f;%f;%d;%d;%d;%d;%d;%d;%d;%d;%s;%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;" % (self.rewardType, self.loadWeightsPath, self.saveWeightsPath, self.learningRate, self.batchSize, str(self.denseLayerUnits), self.numOfBatches, self.bufferSize, self.discountFactor, self.copyWeightsInterval, self.lastEpsilon, self.epsilonDecayRate, self.minimumEpsilon, self.lastLoss, self.cardsPickedCounter, self.okCardCounter, self.notInHandCounter, self.notAllowedCounter, self.requestedNumOfGames, self.gamesCompletedCounter, self.gamesWonCounter, self.gamesLostCounter, str(self.trickScores).replace("[", "").replace("]", ""), str(self.gameScores).replace("[", "").replace("]", ""), self.invalidCardCounters[0][PickCardMode.EXPLORATION], self.invalidCardCounters[1][PickCardMode.EXPLORATION], self.invalidCardCounters[2][PickCardMode.EXPLORATION], self.invalidCardCounters[3][PickCardMode.EXPLORATION], self.invalidCardCounters[4][PickCardMode.EXPLORATION], self.invalidCardCounters[5][PickCardMode.EXPLORATION], self.invalidCardCounters[6][PickCardMode.EXPLORATION], self.invalidCardCounters[7][PickCardMode.EXPLORATION], self.invalidCardCounters[8][PickCardMode.EXPLORATION], self.invalidCardCounters[9][PickCardMode.EXPLORATION], self.invalidCardCounters[10][PickCardMode.EXPLORATION], self.invalidCardCounters[11][PickCardMode.EXPLORATION], self.invalidCardCounters[0][PickCardMode.EXPLOITATION], self.invalidCardCounters[1][PickCardMode.EXPLOITATION], self.invalidCardCounters[2][PickCardMode.EXPLOITATION], self.invalidCardCounters[3][PickCardMode.EXPLOITATION], self.invalidCardCounters[4][PickCardMode.EXPLOITATION], self.invalidCardCounters[5][PickCardMode.EXPLOITATION], self.invalidCardCounters[6][PickCardMode.EXPLOITATION], self.invalidCardCounters[7][PickCardMode.EXPLOITATION], self.invalidCardCounters[8][PickCardMode.EXPLOITATION], self.invalidCardCounters[9][PickCardMode.EXPLOITATION], self.invalidCardCounters[10][PickCardMode.EXPLOITATION], self.invalidCardCounters[11][PickCardMode.EXPLOITATION])
        return r

    def ToLatexValidityTable(self, sessionIndex: int, rewardType: int, includeHeader=True) -> str:
        r = ""
        if includeHeader:
            r += "Experiment \# & Reward \# & Cards picked & OK & NOT IN HAND & NOT ALLOWED \\\\\n\hline"
        rewardType = int(self.rewardType)
        okPercent = "{:.2f}".format(100 * self.okCardCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notInhandPercent = "{:.2f}".format(100 * self.notInHandCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notAllowedPercent = "{:.2f}".format(100 * self.notAllowedCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        r += "\n %d & %d & %d & %d (%s \%%) & %d (%s \%%) & %d (%s \%%) \\\\ \n\hline" % (sessionIndex, rewardType, self.cardsPickedCounter, self.okCardCounter, okPercent, self.notInHandCounter, notInhandPercent, self.notAllowedCounter, notAllowedPercent)
        return r

    def ToLatexWinrateTable(self, sessionIndex: int, rewardType: int, includeHeader=True) -> str:
        r = ""
        if includeHeader:
            r += "Experiment \# & Reward \# & Games played & Games Won & Games lost \\\\\n\hline"
        rewardType = int(self.rewardType)
        wonPercent = "{:.2f}".format(100 * self.gamesWonCounter / self.gamesCompletedCounter) if self.gamesCompletedCounter > 0 else "0.0"
        lostPercent = "{:.2f}".format(100 * self.gamesLostCounter / self.gamesCompletedCounter) if self.gamesCompletedCounter > 0 else "0.0"
        r += "\n %d & %d & %d & %d (%s \%%) & %d (%s \%%) \\\\ \n\hline" % (sessionIndex, rewardType, self.gamesCompletedCounter, self.gamesWonCounter, wonPercent, self.gamesLostCounter, lostPercent)
        return r

    def GetValidityPercentages(self, x_coordinate: str) -> Tuple[str, str, str]:
        okPercent = "{:.2f}".format(100 * self.okCardCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notInHandPercent = "{:.2f}".format(100 * self.notInHandCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notAllowedPercent = "{:.2f}".format(100 * self.notAllowedCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        return ("(%s,%s)" % (x_coordinate, okPercent), "(%s,%s)" % (x_coordinate, notInHandPercent), "(%s,%s)" % (x_coordinate, notAllowedPercent))

    def GetWinratePercentages(self, x_coordinate: str) -> Tuple[str, str]:
        wonPercent = "{:.2f}".format(100 * self.gamesWonCounter / self.gamesCompletedCounter) if self.gamesCompletedCounter > 0 else "0.0"
        lostPercent = "{:.2f}".format(100 * self.gamesLostCounter / self.gamesCompletedCounter) if self.gamesCompletedCounter > 0 else "0.0"
        return ("(%s,%s)" % (x_coordinate, wonPercent), "(%s,%s)" % (x_coordinate, lostPercent))