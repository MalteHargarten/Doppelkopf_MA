import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from doppelkopf.game.Card import Card
from doppelkopf.utils.Console import Console
from doppelkopf.game.GameState import GameState
from doppelkopf.models.LSTMModel import LSTMModel
from doppelkopf.data.LSTMSupervisedDataset import LSTMSupervisedDataset

class LSTMSupervisedTrainer():
    def __init__(self, name:str, batchSize: int, learningRate: float, savedWeightsPath=None) -> None:
        self.name = name
        self.batchSize = batchSize
        self.lstm = LSTMModel.Create("Supervised LSTM", tf.float32, GameState.SIZE_STATE, [], Card.NUM_CARDTYPES, learningRate, savedWeightsPath)

    def Train(self, dataset: LSTMSupervisedDataset, epochs: int, loadSize: int, weightsPath: str):
        opened = dataset.Open()
        Console.WriteSuccess("Found a set with %d samples from file!" % dataset.Size(), self.name)
        for i in range(epochs):
            dataset.ResetLoadStart()
            Console.WriteSuccess("Epoch %d/%d" % (i+1, epochs), self.name)
            while dataset.HasNextLoad():
                batchX, batchY = dataset.NextLoad(loadSize, shuffle=True, cacheResult=True)
                Console.WriteSuccess("x_batch.shape: %s" % str(batchX.shape), self.name)
                Console.WriteSuccess("y_batch.shape: %s" % str(batchY.shape), self.name)
                self.lstm.fit(x=batchX, y=batchY, batch_size=self.batchSize, epochs=1)
        if opened:
            dataset.Close()
        self.lstm.TrySaveWeights(weightsPath)
        Console.WriteSuccess("Training completed", self.name)

    def Evaluate(self, evaluationSet: LSTMSupervisedDataset, loadSize: int):
        evaluationSet.Open()
        Console.WriteSuccess("Found a set with %d samples from file!" % evaluationSet.Size(), self.name)
        evaluationSet.ResetLoadStart()
        error_values = []
        while evaluationSet.HasNextLoad():
            x_batch, y_batch = evaluationSet.NextLoad(loadSize, shuffle=True, cacheResult=True)
            Console.WriteSuccess("x_batch.shape: %s" % str(x_batch.shape), self.name)
            Console.WriteSuccess("y_batch.shape: %s" % str(y_batch.shape), self.name)
            error_values.append(self.lstm.evaluate(x=x_batch, y=y_batch, batch_size=self.batchSize))
        evaluationSet.Close()
        averageError = np.average(np.array(error_values))
        Console.WriteSuccess("Evaluation complete. Average error is: %f" % averageError, self.name)
        return averageError