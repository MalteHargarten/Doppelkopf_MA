import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.agents.LSTMPlayer import LSTMPlayer
from doppelkopf.programs.Program import Program
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunLSTMPlayer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue='localhost'),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=None),
            OptionalArgument(name="dnnWeightsPath", expectedType=str, defaultValue='../../../models/DNN/trainedModel'),
            OptionalArgument(name="lstmWeightsPath", expectedType=str, defaultValue='../../../models/LSTM/RL/per-trick-reward/trainedModel'),
            OptionalArgument(name="denseLayerUnits", expectedType=list, defaultValue=[256, 128, 64]),
            OptionalArgument(name="hiddenLSTMUnits", expectedType=list, defaultValue=[]),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
        ]
        super(RunLSTMPlayer, self).__init__([], optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        numOfGames = self.GetArgumentByName("numOfGames")
        dnnWeightsPath = self.GetArgumentByName("dnnWeightsPath")
        lstmWeightsPath = self.GetArgumentByName("lstmWeightsPath")
        denseLayerUnits = self.GetArgumentByName("denseLayerUnits")
        hiddenLSTMUnits = self.GetArgumentByName("hiddenLSTMUnits")
        logFile = self.GetArgumentByName("logFile")
        # # # # # # # # # # # # # # # Run DNN Agent # # # # # # # # # # # # # # #
        player = LSTMPlayer(dnnWeightsPath, lstmWeightsPath, denseLayerUnits, hiddenLSTMUnits)
        player.ConnectToServer(host, port) # Connect to Server
        player.PlayGames(numOfGames)
        if logFile is not None:
            player.LogReport(logFile, numOfGames)
        player.DisconnectFromServer() # Disconnect from Server

def main(args):
    program = RunLSTMPlayer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)