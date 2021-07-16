from doppelkopf.agents.EnumsRL import PickCardMode
from doppelkopf.reports.DNNPlayReport import DNNPlayReport

# Note: This class is not currently used anywhere, due to LSTM research being discontinued!
class LSTMPlayReport(DNNPlayReport):
    def __init__(self, datetime: str, dnnWeightsPath: str, weightsPath: str, requestedNumOfGames: int, cardPickedCounter: int, okCardCounter: int, notInHandCounter: int, notAllowedCounter: int, gamesCompletedCounter: int, gamesWonCounter: int, gamesLostCounter: int, invalidCardCounters: dict):
        super().__init__(datetime, weightsPath, requestedNumOfGames, cardPickedCounter, okCardCounter, notInHandCounter, notAllowedCounter, gamesCompletedCounter, gamesWonCounter, gamesLostCounter, invalidCardCounters)
        self.dnnWeightsPath = dnnWeightsPath

    def __str__(self) -> str:
        r = "Play Session recorded on %s" % (self.datetime)
        r += "\nDNN used weights from '%s', while the LSTM used the weights from %s" % (self.loadWeightsPath, self.dnnWeightsPath)
        if self.gamesCompletedCounter > 0:
            if self.requestedNumOfGames is not None:
                r += "\n%d/%d games were completed successfully!" % (self.gamesCompletedCounter, self.requestedNumOfGames)
            else:
                r += "\n%d games were completed successfully!" % (self.gamesCompletedCounter)
            r += "\n%d/%d (%f %%) games were won!" % (self.gamesWonCounter, self.gamesCompletedCounter, (self.gamesWonCounter / self.gamesCompletedCounter))
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