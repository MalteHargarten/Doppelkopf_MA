import sys
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.data.DNNDataset import DNNDataset
from doppelkopf.programs.Argument import Argument
from doppelkopf.agents.DNNTrainerSL import DNNTrainerSL
from doppelkopf.programs.OptionalArgument import OptionalArgument

class RunDNNTrainerSL(Program):
    def __init__(self):
        required = [
            Argument(name="saveWeightsPath", expectedType=str),
        ]
        optionals = [
            OptionalArgument(name="epochs", expectedType=int, defaultValue=25),
            OptionalArgument(name="learningRate", expectedType=float, defaultValue=0.001),
            OptionalArgument(name="loadSize", expectedType=int, defaultValue=500000),
            OptionalArgument(name="cacheSize", expectedType=int, defaultValue=500000),
            OptionalArgument(name="batchSize", expectedType=int, defaultValue=100),
            OptionalArgument(name="trainPath", expectedType=str, defaultValue="../../../data/DNN_supervised_training"),
            OptionalArgument(name="evaluationPath", expectedType=str, defaultValue="../../../data/DNN_supervised_evaluation"),
            OptionalArgument(name="loadWeightsPath", expectedType=str, defaultValue=None),
            OptionalArgument(name="denseLayerUnits", expectedType=list, defaultValue=[256, 128, 64]),
            OptionalArgument(name="logFile", expectedType=str, defaultValue=None),
        ]
        super(RunDNNTrainerSL, self).__init__(required, optionals)

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        epochs = self.GetArgumentByName("epochs")
        learningRate = self.GetArgumentByName("learningRate")
        loadSize = self.GetArgumentByName("loadSize")
        cacheSize = self.GetArgumentByName("cacheSize")
        batchSize = self.GetArgumentByName("batchSize")
        trainPath = self.GetArgumentByName("trainPath")
        evaluationPath = self.GetArgumentByName("evaluationPath")
        loadWeightsPath = self.GetArgumentByName("loadWeightsPath")
        saveWeightsPath = self.GetArgumentByName("saveWeightsPath")
        denseLayerUnits = self.GetArgumentByName("denseLayerUnits")
        logFile = self.GetArgumentByName("logFile")
        # # # # # # # # # # # # # # # Train and evaluate agent # # # # # # # # # # # # # # #
        supervisedTrainer = DNNTrainerSL("DNN_Model", learningRate, loadSize, batchSize, denseLayerUnits, loadWeightsPath) # Try loading weights (if they already exist)
        trainingsSet = DNNDataset(trainPath, cacheSize=cacheSize)
        trainingsSet.Open()
        supervisedTrainer.Train(epochs, trainingsSet, saveWeightsPath)
        evalLoss = supervisedTrainer.Evaluate(evaluationPath, cacheSize)
        if logFile is not None:
            supervisedTrainer.LogReport(logFile, epochs, trainingsSet.Size(), evalLoss, saveWeightsPath)
        trainingsSet.Close()

def main(args):
    program = RunDNNTrainerSL()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)