import sys
from doppelkopf.programs.Program import Program
from doppelkopf.agents.DNNDataRecorderSL import DNNDataRecorderSL
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunDNNDataRecorderSL(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue='localhost'),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=10),
            OptionalArgument(name="percentage", expectedType=float, defaultValue=0.8),
            OptionalArgument(name="trainingPath", expectedType=str, defaultValue="../../../data/DNN/training"),
            OptionalArgument(name="evaluationPath", expectedType=str, defaultValue="../../../data/DNN/evaluation"),
        ]
        super(RunDNNDataRecorderSL, self).__init__([], optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        host = self.GetArgumentByName("host")
        port = self.GetArgumentByName("port")
        numOfGames = self.GetArgumentByName("numOfGames")
        percentage = self.GetArgumentByName("percentage")
        trainingPath = self.GetArgumentByName("trainingPath")
        evaluationPath = self.GetArgumentByName("evaluationPath")
        # # # # # # # # # # # # # # # Check special conditions # # # # # # # # # # # # # # #
        if numOfGames <= 0:
            exit("Parameter 'numOfGames' was set to 0 (zero). Exiting...")
        if percentage < 0.0 or percentage > 1.0:
            exit("Parameter 'percentage' (%f) was out of bounds [0.0-1.0]" % percentage)
        # # # # # # # # # # # # # # # Proceed if parameters check out # # # # # # # # # # # # # # #
        trainingGames = int(numOfGames * percentage)
        evaluationGames = numOfGames - trainingGames
        # # # # # # # # # # # # # # # Training data # # # # # # # # # # # # # # #
        if trainingGames > 0:
            trainingRecorder = DNNDataRecorderSL(trainingPath)
            trainingRecorder.ConnectToServer(host, port)
            trainingRecorder.PlayGames(trainingGames)
            trainingRecorder.DisconnectFromServer()
        # # # # # # # # # # # # # # # Evaluation data # # # # # # # # # # # # # # #
        if evaluationGames > 0:
            evaluationRecorder = DNNDataRecorderSL(evaluationPath)
            evaluationRecorder.ConnectToServer(host, port)
            evaluationRecorder.PlayGames(evaluationGames)
            evaluationRecorder.DisconnectFromServer()

def main(args):
    program = RunDNNDataRecorderSL()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)