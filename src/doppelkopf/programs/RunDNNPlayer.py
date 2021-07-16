import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.agents.DNNPlayer import DNNPlayer
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunDNNPlayer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue='localhost'),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=None),
            OptionalArgument(name="learningRate", expectedType=float, defaultValue=0.001),
            OptionalArgument(name="loadWeightsPath", expectedType=str, defaultValue=None),
            OptionalArgument(name="denseLayerUnits", expectedType=list, defaultValue=[256, 128, 64]),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
        ]
        super(RunDNNPlayer, self).__init__([], optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        numOfGames = self.GetArgumentByName("numOfGames")
        learningRate = self.GetArgumentByName("learningRate")
        loadWeightsPath = self.GetArgumentByName("loadWeightsPath")
        denseLayerUnits = self.GetArgumentByName("denseLayerUnits")
        logFile = self.GetArgumentByName("logFile")
        # # # # # # # # # # # # # # # Run DNN Agent # # # # # # # # # # # # # # #
        player = DNNPlayer(learningRate, loadWeightsPath, denseLayerUnits) # Use default parameters for learningRate and batchSize because this script does not perform any training anyway
        player.ConnectToServer(host, port) # Connect to Server
        player.PlayGames(numOfGames)
        if logFile is not None:
            player.LogReport(logFile, numOfGames)
        player.DisconnectFromServer() # Disconnect from Server

def main(args):
    program = RunDNNPlayer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)