import sys
import os
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
from typing import List
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.programs.Argument import Argument
from doppelkopf.agents.DNNTrainerRL import RewardType
from doppelkopf.agents.DNNTrainerRL import DNNTrainerRL
from doppelkopf.agents.RulebasedPlayer import RulebasedPlayer
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunDNNTrainerRL(Program):
    def __init__(self):
        required = [
            Argument(name="saveWeightsPath", expectedType=str)
        ]
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue="localhost"),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=None),
            OptionalArgument(name="learningRate", expectedType=float, defaultValue=0.001),
            OptionalArgument(name="loadWeightsPath", expectedType=str, defaultValue=None),
            OptionalArgument(name="denseLayerUnits", expectedType=list, defaultValue=[256, 128, 64]),
            OptionalArgument(name="copyWeightsInterval", expectedType=int, defaultValue=100),
            OptionalArgument(name="bufferSize", expectedType=int, defaultValue=10000),
            OptionalArgument(name="batchSize", expectedType=int, defaultValue=100),
            OptionalArgument(name="numOfBatches", expectedType=int, defaultValue=20),
            OptionalArgument(name="discountFactor", expectedType=float, defaultValue=0.1),
            OptionalArgument(name="epsilon", expectedType=float, defaultValue=1.0),
            OptionalArgument(name="epsilonDecayRate", expectedType=float, defaultValue=0.997),
            OptionalArgument(name="minimumEpsilon", expectedType=float, defaultValue=0.05),
            OptionalArgument(name="rewardType", expectedType=RewardType, defaultValue=RewardType.PER_VALID_CARD),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
        ]
        super(RunDNNTrainerRL, self).__init__(required, optionals) # Call base constructor

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        numOfGames = self.GetArgumentByName("numOfGames")
        learningRate = self.GetArgumentByName("learningRate")
        loadWeightsPath = self.GetArgumentByName("loadWeightsPath")
        saveWeightsPath = self.GetArgumentByName("saveWeightsPath")
        denseLayerUnits = self.GetArgumentByName("denseLayerUnits")
        copyWeightsInterval = self.GetArgumentByName("copyWeightsInterval")
        bufferSize = self.GetArgumentByName("bufferSize")
        batchSize = self.GetArgumentByName("batchSize")
        numOfBatches = self.GetArgumentByName("numOfBatches")
        discountFactor = self.GetArgumentByName("discountFactor")
        epsilon = self.GetArgumentByName("epsilon")
        epsilonDecayRate = self.GetArgumentByName("epsilonDecayRate")
        minimumEpsilon = self.GetArgumentByName("minimumEpsilon")
        rewardType = self.GetArgumentByName("rewardType")
        logFile = self.GetArgumentByName("logFile")
        # # # # # # # # # # # # # # # Create agent # # # # # # # # # # # # # # #
        trainer = DNNTrainerRL(
            learningRate,
            loadWeightsPath, # If there already is a trained model, load it
            saveWeightsPath,
            denseLayerUnits,
            copyWeightsInterval,
            bufferSize,
            batchSize,
            numOfBatches,
            discountFactor,
            epsilon,
            epsilonDecayRate,
            minimumEpsilon,
            rewardType
        )
        # # # # # # # # # # # # # # # Start Dummy players in threads # # # # # # # # # # # # # # #
        dummyPlayers: List[RulebasedPlayer] = []
        dummyPlayerThreads: List[threading.Thread] = []
        for i in range(3):
            dummyPlayers.append(RulebasedPlayer())
            dummyPlayerThreads.append(threading.Thread(target=self.threadRunDummyPlayer, args=(dummyPlayers[i], host, port, numOfGames)))
            dummyPlayerThreads[i].start()
        # # # # # # # # # # # # # # # Train the agent # # # # # # # # # # # # # # #
        trainer.ConnectToServer(host, port)
        lastloss = trainer.DoReinforcementLearning(numOfGames)
        if logFile is not None:
            trainer.LogReport(logFile, numOfGames, lastloss)
        trainer.DisconnectFromServer()
        # # # # # # # # # # # # # # # End Dummy player threads # # # # # # # # # # # # # # #
        for i in range(3):
            dummyPlayers[i].client.Stop()
            dummyPlayerThreads[i].join()

    def threadRunDummyPlayer(self, dummyPlayer: RulebasedPlayer, host: str, port: int, numOfGames):
        dummyPlayer.ConnectToServer(host, port)
        dummyPlayer.PlayGames(numOfGames, canBeInterrupted=False)
        dummyPlayer.DisconnectFromServer()

def main(args):
    program = RunDNNTrainerRL()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)