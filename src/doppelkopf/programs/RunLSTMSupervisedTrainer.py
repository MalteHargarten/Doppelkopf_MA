from json import loads
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.programs.Program import Program
from doppelkopf.programs.OptionalArgument import OptionalArgument
from doppelkopf.data.LSTMSupervisedDataset import LSTMSupervisedDataset
from doppelkopf.agents.LSTMSupervisedTrainer import LSTMSupervisedTrainer

class RunLSTMSupervisedTrainer(Program):
    def __init__(self):
        optionals = [
            OptionalArgument(name="epochs", expectedType=int, defaultValue=25),
            OptionalArgument(name="learningRate", expectedType=float, defaultValue=0.001),
            OptionalArgument(name="loadSize", expectedType=int, defaultValue=500000),
            OptionalArgument(name="cacheSize", expectedType=int, defaultValue=500000),
            OptionalArgument(name="batchSize", expectedType=int, defaultValue=2),
            OptionalArgument(name="trainingDataPath", expectedType=str, defaultValue="../../../data/LSTM/supervisedTraining"),
            OptionalArgument(name="evaluationDataPath", expectedType=str, defaultValue="../../../data/LSTM/supervisedEvaluation"),
            OptionalArgument(name="lstmWeightsPath", expectedType=str, defaultValue="../../../models/LSTM/supervised/supervisedModel"),
        ]
        super(RunLSTMSupervisedTrainer, self).__init__([], optionals) # Call base constructor

    def onRun(self):
        # # # # # # # # # # # # # # # Get values from parameters # # # # # # # # # # # # # # #
        epochs = self.GetArgumentByName("epochs")
        learningRate = self.GetArgumentByName("learningRate")
        loadSize = self.GetArgumentByName("loadSize")
        cacheSize = self.GetArgumentByName("cacheSize")
        batchSize = self.GetArgumentByName("batchSize")
        trainingPath = self.GetArgumentByName("trainingDataPath")
        evaluationPath = self.GetArgumentByName("evaluationDataPath")
        lstmWeightsPath = self.GetArgumentByName("lstmWeightsPath")
        # # # # # # # # # # # # # # # Create agent # # # # # # # # # # # # # # #
        trainer = LSTMSupervisedTrainer("Trainer", batchSize, learningRate, lstmWeightsPath) # Try loading weights (if they already exist)
        # # # # # # # # # # # # # # # Train the agent # # # # # # # # # # # # # # #
        trainSet = LSTMSupervisedDataset(trainingPath, cacheSize=cacheSize)
        trainer.Train(trainSet, epochs, loadSize, lstmWeightsPath)
        evalSet = LSTMSupervisedDataset(evaluationPath, cacheSize=cacheSize)
        trainer.Evaluate(evalSet, cacheSize)

def main(args):
    program = RunLSTMSupervisedTrainer()
    program.Run(args)

if __name__ == '__main__':
    main(sys.argv)