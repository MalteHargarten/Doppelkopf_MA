import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.programs.OptionalArgument import OptionalArgument
from doppelkopf.agents.LSTMSupervisedPlayer import LSTMSupervisedPlayer

class RunLSTMPlayer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue='localhost'),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=None),
            OptionalArgument(name="lstmWeightsPath", expectedType=str, defaultValue='../../../models/LSTM/supervised/supervisedModel'),
        ]
        super(RunLSTMPlayer, self).__init__([], optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName('host')
        port = self.GetArgumentByName('port')
        numOfGames = self.GetArgumentByName('numOfGames')
        lstmWeightsPath = self.GetArgumentByName('lstmWeightsPath')
        # # # # # # # # # # # # # # # Run DNN Agent # # # # # # # # # # # # # # #
        player = LSTMSupervisedPlayer(loadWeightsPath=lstmWeightsPath) # Use default parameters for learningRate and batchSize because this script does not perform any training anyway
        player.ConnectToServer(host, port) # Connect to Server
        player.PlayGames(numOfGames)
        player.DisconnectFromServer() # Disconnect from Server

def main(args):
    program = RunLSTMPlayer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)