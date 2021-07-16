import numpy as np
from typing import List
from doppelkopf.models.DNN import DNN
from doppelkopf.utils.File import File
from doppelkopf.utils.Helper import Helper
from doppelkopf.utils.Console import Console
from doppelkopf.data.DNNDataset import DNNDataset
from doppelkopf.reports.DNNTrainReportSL import DNNTrainReportSL

class DNNTrainerSL():
    def __init__(self, dnnName: str, learningRate, loadSize, batchSize, denseLayerUnits: List[int], loadWeightsPath: str):
        self.dnnName = dnnName
        self.learningRate = learningRate
        self.loadSize = loadSize
        self.batchSize = batchSize
        self.loadWeightsPath = loadWeightsPath
        self.model = DNN(self.dnnName, self.learningRate, denseLayerUnits, loadWeightsPath=loadWeightsPath)
        self.lastTrainingLoss = None

    def Train(self, epochs: int, trainingsSet: DNNDataset, saveWeightsPath = None):
        opened = trainingsSet.Open()
        Console.WriteInfo("Found a set with %d samples from file!" % trainingsSet.Size(), self.dnnName)
        if self.loadSize >= trainingsSet.Size(): # If we plan on loading all samples at once
            trainingsSet.ResetLoadStart()
            Console.WriteInfo("Loading %d samples, this might take a while..." % (self.loadSize), self.dnnName)
            x, y = trainingsSet.NextLoad(self.loadSize, shuffle=True, cacheResult=True)
            self.lastTrainingLoss = self.model.fit(x=x, y=y, batch_size=self.batchSize, epochs=epochs).history["loss"][-1] # Get the history's last loss
        else:
            for i in range(epochs):
                trainingsSet.ResetLoadStart()
                Console.WriteInfo("Epoch %d/%d" % (i+1, epochs), self.dnnName)
                while trainingsSet.HasNextLoad():
                    Console.WriteInfo("Loading %d samples, this might take a while..." % (self.loadSize), self.dnnName)
                    x, y = trainingsSet.NextLoad(self.loadSize, shuffle=True, cacheResult=True)
                    Console.WriteInfo("x.shape: %s" % str(x.shape), self.dnnName)
                    Console.WriteInfo("y.shape: %s" % str(y.shape), self.dnnName)
                    self.lastTrainingLoss = self.model.fit(x=x, y=y, batch_size=self.batchSize, epochs=1).history["loss"][-1] # Get the history's last loss
        if opened:
            trainingsSet.Close()
        if saveWeightsPath is not None:
            self.model.TrySaveWeights(saveWeightsPath)
        Console.WriteSuccess("Training completed", self.dnnName)

    def Evaluate(self, datapath: str, cacheSize: int):
        evaluationSet = DNNDataset(datapath, cacheSize=cacheSize)
        evaluationSet.Open()
        Console.WriteInfo("Found a set with %d samples from file!" % evaluationSet.Size(), self.dnnName)
        evaluationSet.ResetLoadStart()
        error_values = []
        while evaluationSet.HasNextLoad():
            Console.WriteInfo("Loading %d samples, this might take a while..." % (self.loadSize), self.dnnName)
            x, y = evaluationSet.NextLoad(self.loadSize, shuffle=True, cacheResult=True)
            Console.WriteInfo("x.shape: %s" % str(x.shape), self.dnnName)
            Console.WriteInfo("y.shape: %s" % str(y.shape), self.dnnName)
            error_values.append(self.model.evaluate(x=x, y=y, batch_size=self.batchSize))
        evaluationSet.Close()
        averageError = np.average(np.array(error_values))
        Console.WriteSuccess("Evaluation complete. Average error is: %f" % averageError, self.dnnName)
        return averageError

    def LogReport(self, logFile: str, epochs: int, sampleSize: int, evaluationLoss: float, saveWeightsPath: str):
        report = DNNTrainReportSL(Helper.DateTimeNowToString(), self.learningRate, epochs, sampleSize, self.batchSize, self.loadWeightsPath, saveWeightsPath, self.lastTrainingLoss, evaluationLoss)
        File.Append(report, logFile) # Write result object to file