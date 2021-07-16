import sys
from doppelkopf.programs.Program import Program
from doppelkopf.programs.OptionalArgument import OptionalArgument
from doppelkopf.agents.LSTMSupervisedRecorder import LSTMSupervisedRecorder

class RunLSTMSupervisedRecorder(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="host", expectedType=str, defaultValue='localhost'),
            OptionalArgument(name="port", expectedType=int, defaultValue=8088),
            OptionalArgument(name="numOfGames", expectedType=int, defaultValue=10),
            OptionalArgument(name="percentage", expectedType=float, defaultValue=0.8),
            OptionalArgument(name="trainingPath", expectedType=str, defaultValue="../../../data/LSTM/supervisedTraining"),
            OptionalArgument(name="evaluationPath", expectedType=str, defaultValue="../../../data/DNN/supervisedEvaluation"),
        ]
        super(RunLSTMSupervisedRecorder, self).__init__([], optionals)

    def onRun(self):
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
            trainingRecorder = LSTMSupervisedRecorder(trainingPath, 500000)
            trainingRecorder.ConnectToServer(host, port)
            trainingRecorder.PlayGames(trainingGames)
            trainingRecorder.DisconnectFromServer()
        # # # # # # # # # # # # # # # Evaluation data # # # # # # # # # # # # # # #
        if evaluationGames > 0:
            evaluationRecorder = LSTMSupervisedRecorder(evaluationPath, 500000)
            evaluationRecorder.ConnectToServer(host, port)
            evaluationRecorder.PlayGames(evaluationGames)
            evaluationRecorder.DisconnectFromServer()

def main(args):
    program = RunLSTMSupervisedRecorder()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)