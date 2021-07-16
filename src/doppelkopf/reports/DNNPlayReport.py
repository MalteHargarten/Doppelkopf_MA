import numpy as np
from typing import List
from doppelkopf.agents.EnumsRL import PickCardMode
from doppelkopf.reports.PlayReport import PlayReport

class DNNPlayReport(PlayReport):
    def __init__(self, datetime: str, loadWeightsPath: str, requestedNumOfGames: int, cardsPickedCounter: int, okCardCounter: int, notInHandCounter: int, notAllowedCounter: int, gamesCompletedCounter: int, gamesWonCounter: int, gamesLostCounter: int, trickScores: List[int], gameScores: List[int], invalidCardCounters: dict):
        super(DNNPlayReport, self).__init__(datetime, requestedNumOfGames, cardsPickedCounter, okCardCounter, notInHandCounter, notAllowedCounter, gamesCompletedCounter, gamesWonCounter, gamesLostCounter, trickScores, gameScores)
        self.loadWeightsPath = loadWeightsPath
        self.invalidCardCounters = invalidCardCounters

    def __str__(self) -> str:
        r = "Play Session recorded on %s" % (self.datetime)
        r += "\nDNN used weights from '%s'" % (self.loadWeightsPath)
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
            r += "Weights Path;Total cards picked;Cards OK;Cards NOT_IN_HAND;Cards NOT_ALLOWED;Requested Games;Completed Games;Games Won;Games Lost;Trick Scores;GameScores;Wrong Cards Trick 1;Wrong Cards Trick 2;Wrong Cards Trick 3;Wrong Cards Trick 4;Wrong Cards Trick 5;Wrong Cards Trick 6;Wrong Cards Trick 7;Wrong Cards Trick 8;Wrong Cards Trick 9;Wrong Cards Trick 10;Wrong Cards Trick 11;Wrong Cards Trick 12\n"
        r += "%s;%d;%d;%d;%d;%d;%d;%d;%d;%s;%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;" % (self.loadWeightsPath, self.cardsPickedCounter, self.okCardCounter, self.notInHandCounter, self.notAllowedCounter, self.requestedNumOfGames, self.gamesCompletedCounter, self.gamesWonCounter, self.gamesLostCounter, str(self.trickScores).replace("[", "").replace("]", ""), str(self.gameScores).replace("[", "").replace("]", ""), self.invalidCardCounters[0], self.invalidCardCounters[1], self.invalidCardCounters[2], self.invalidCardCounters[3], self.invalidCardCounters[4], self.invalidCardCounters[5], self.invalidCardCounters[6], self.invalidCardCounters[7], self.invalidCardCounters[8], self.invalidCardCounters[9], self.invalidCardCounters[10], self.invalidCardCounters[11])
        return r
