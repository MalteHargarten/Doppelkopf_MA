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

class DNN(tf.keras.Sequential):
    def __init__(self, name, learningRate, denseLayerUnits, inputSize=GameState.SIZE_STATE, outputSize=Card.NUM_CARDTYPES, loadWeightsPath=None):
        super(DNN, self).__init__(name=name)
        self.add(tf.keras.layers.InputLayer(input_shape=(inputSize), name="InputGameState"))
        for i, denseLayerUnit in enumerate(denseLayerUnits):
            self.add(tf.keras.layers.Dense(units=denseLayerUnit, activation=tf.keras.activations.relu, name="Dense%d" % i))
        self.add(tf.keras.layers.Dense(units=outputSize, name="Output")) # Last layer has no activation
        self.myOptimiser = tf.keras.optimizers.Adam(learning_rate=learningRate)
        self.compile(self.myOptimiser, loss=tf.keras.losses.mean_squared_error)
        Console.WriteSuccess("Compiled", "DNN %s" % name)
        self.summary()
        if loadWeightsPath is not None:
            self.TryLoadWeights(loadWeightsPath)

    def CopyWeightsFrom(self, otherLSTMModel: tf.keras.Model):
        self.set_weights(otherLSTMModel.get_weights())
        Console.WriteInfo("Weights copied from %s to %s" % (otherLSTMModel.name, self.name), self.name)

    def TryLoadWeights(self, loadWeightsPath):
        try:
            status = self.load_weights(loadWeightsPath)
            Console.WriteSuccess("Weights loaded from %s" % loadWeightsPath, self.name)
            return True
        except Exception as e:
            Console.WriteError(e, self.name)
            return False

    def TrySaveWeights(self, weightsPath):
        try:
            self.save_weights(weightsPath)
            Console.WriteSuccess("Weights saved to %s" % weightsPath, self.name)
            return True
        except Exception as e:
            Console.WriteError(e, self.name)
            return False

    def QValues(self, inputs: np.ndarray):
        return self(inputs)

    def QValuesArgmax(self, inputs: np.ndarray):
        output = self.QValues(inputs)
        return tf.argmax(output, axis=2)

    def QValueAt(self, index, inputs: np.ndarray):
        output = self.QValues(inputs)
        return output[:,:,index]

    def QValuesNumpy(self, inputs: np.ndarray) -> np.ndarray:
        return tf.keras.backend.eval(self.QValues(inputs)) # Evaluate the tensor to get its values

    def QValuesArgmaxNumpy(self, inputs: np.ndarray) -> int:
        return tf.keras.backend.eval(self.QValuesArgmax(inputs))

    def QValueAtNumpy(self, index, inputs: np.ndarray) -> np.ndarray:
        return tf.keras.backend.eval(self.QValueAt(index, inputs))

    def ApplyGradients(self, gradients):
        self.myOptimiser.apply_gradients(zip(gradients, self.trainable_variables))