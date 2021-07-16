import numpy as np
from typing import List, Tuple

class PlayReport():
    def __init__(self, datetime: str, requestedNumOfGames: int, cardsPickedCounter: int, okCardCounter: int, notInHandCounter: int, notAllowedCounter: int, gamesCompletedCounter: int, gamesWonCounter: int, gamesLostCounter: int, trickScores: List[int], gameScores: List[int]):
        self.datetime = datetime
        self.requestedNumOfGames = requestedNumOfGames
        self.cardsPickedCounter = cardsPickedCounter
        self.okCardCounter = okCardCounter
        self.notInHandCounter = notInHandCounter
        self.notAllowedCounter = notAllowedCounter
        self.gamesCompletedCounter = gamesCompletedCounter
        self.gamesWonCounter = gamesWonCounter
        self.gamesLostCounter = gamesLostCounter
        self.trickScores = trickScores
        self.gameScores = gameScores
    
    def __str__(self) -> str:
        r = "Play Session recorded on %s" % (self.datetime)
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
        return r

    def ToCSV(self, includeHeader=True) -> str:
        r = ""
        if includeHeader:
            r += "Total cards picked;Cards OK;Cards NOT_IN_HAND;Cards NOT_ALLOWED;Requested Games;Completed Games;Games Won;Games Lost;Trick Scores;Game Scores;\n"
        r += "%d;%d;%d;%d;%d;%d;%d;%d;%s;%s" % (self.cardsPickedCounter, self.okCardCounter, self.notInHandCounter, self.notAllowedCounter, self.requestedNumOfGames, self.gamesCompletedCounter, self.gamesWonCounter, self.gamesLostCounter, str(self.trickScores).replace("[", "").replace("]", ""), str(self.gameScores).replace("[", "").replace("]", ""))
        return r

    def ToLatexValidityTable(self, sessionIndex: int, rewardType: int, includeHeader=True) -> str:
        r = ""
        if includeHeader:
            r += "Experiment \# & Reward \# & Cards picked & OK & NOT IN HAND & NOT ALLOWED \\\\\n\hline"
        okPercent = "{:.2f}".format(100 * self.okCardCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notInHandPercent = "{:.2f}".format(100 * self.notInHandCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        notAllowedPercent = "{:.2f}".format(100 * self.notAllowedCounter / self.cardsPickedCounter) if self.cardsPickedCounter > 0 else "0.0"
        r += "\n %d & %d & %d & %d (%s \%%) & %d (%s \%%) & %d (%s \%%) \\\\ \n\hline" % (sessionIndex, rewardType, self.cardsPickedCounter, self.okCardCounter, okPercent, self.notInHandCounter, notInHandPercent, self.notAllowedCounter, notAllowedPercent)
        return r

    def ToLatexWinrateTable(self, sessionIndex: int, rewardType: int, includeHeader=True) -> str:
        r = ""
        if includeHeader:
            r += "Experiment \# & Reward \# & Games played & Games Won & Games lost \\\\\n\hline"
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