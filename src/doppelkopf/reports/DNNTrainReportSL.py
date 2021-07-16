from typing import List

class DNNTrainReportSL():
    def __init__(self, datetime: str, learningRate: float, epochs: int, totalNumOfSamples: int, batchSize: int, loadWeightsPath: str, saveWeightsPath: str, lastTrainingLoss: float, evaluationLoss: float):
        self.datetime = datetime
        self.learningRate = learningRate
        self.epochs = epochs
        self.totalNumOfSamples = totalNumOfSamples
        self.batchSize = batchSize
        self.loadWeightsPath = loadWeightsPath
        self.saveWeightsPath = saveWeightsPath
        self.lastTrainingLoss = lastTrainingLoss
        self.evaluationLoss = evaluationLoss

    def __str__(self) -> str:
        r = "Training Session recorded on %s" % (self.datetime)
        r += "\nDNN loaded weights from '%s' and saved them to %s, a learning_rate of %f" % (self.loadWeightsPath, self.saveWeightsPath, self.learningRate)
        r += "\nThe DNN Trainer trained for %d epochs, using %d samples, with %d samples per batch" % (self.epochs, self.totalNumOfSamples, self.batchSize)
        #r += "\nThe DNN Trainer trained for %d epochs, with %d samples per batch" % (self.epochs, self.batchSize)
        r += "\nThe last training loss value on record is %f, while the evaluation loss was %f" % (self.lastTrainingLoss, self.evaluationLoss)
        return r

    def ToCSV(self, includeHeader=True):
        r = ""
        if includeHeader:
            r += "Load Weights Path;Save Weights Path;Learning Rate;Epochs;Total Number of Samples;Batch Size;Last Training Loss;Evaluation Loss\n"
        r += "%s;%s;%f;%d;%d;%d;%f;%f" % (self.loadWeightsPath, self.saveWeightsPath, self.learningRate, self.epochs, self.totalNumOfSamples, self.batchSize, self.lastTrainingLoss, self.evaluationLoss)
        return r