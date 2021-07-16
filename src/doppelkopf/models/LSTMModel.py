import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # 1 = Silence INFO messages, 2 = Silence INFO and Warning messages
import numpy as np
import tensorflow as tf
from typing import List
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from doppelkopf.utils.Console import Console
from doppelkopf.models.LSTMLayer import LSTMLayer
from doppelkopf.game.GameState import GameState
from doppelkopf.game.Card import Card

class LSTMModel(tf.keras.Model):
    @staticmethod
    def Create(name: str, inputType, inputSize: int, hiddenLSTMUnits: List[int], outputUnits: int, learningRate: float, savedWeightsPath=None):
        # # # # # # # # # # # # # # # Create layers # # # # # # # # # # # # # # #
        inputTensor = tf.keras.layers.Input(shape=(None, inputSize), name="LSTMInputLayer", dtype=inputType)
        lstmLayers = []
        for i, unithape in enumerate(hiddenLSTMUnits):
            lstmLayers.append(LSTMLayer(units=unithape, name="LSTMLayer_%d" % i))
        lstmLayers.append(LSTMLayer(outputUnits, "LSTMOutput"))
        # # # # # # # # # # # # # # # Gather all the outputs # # # # # # # # # # # # # # #
        #outputs = []
        out = inputTensor
        #hs = []
        #cs = []
        for lstmLayer in lstmLayers:
            out, h, c = lstmLayer(out)
            #hs.append(h)
            #cs.append(c)
        #outputs.append(out)
        #outputs.append(hs)
        #utputs.append(cs)
        inputs = [inputTensor]
        outputs = [out]
        return LSTMModel(name, inputs, outputs, inputSize, outputUnits, lstmLayers, learningRate, savedWeightsPath)

    def __init__(self, name, inputs: list, outputs: list, inputSize: int, outputUnits: int, lstmLayers: List[LSTMLayer], learningRate, savedWeightsPath=None):
        # # # # # # # # # # # # # # # Create and compile model (self) # # # # # # # # # # # # # # #
        super(LSTMModel, self).__init__(inputs=inputs, outputs=outputs, name="LSTMModel %s" % name) # Create model using these inputs and outputs
        self.lstmLayers = lstmLayers
        self.inputSize = inputSize
        self.outputUnits = outputUnits
        self.learningRate = learningRate
        self.myOptimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        self.compile(optimizer=self.myOptimizer, loss=tf.keras.losses.mean_squared_error, run_eagerly=True) # Should be enabled by default #tf.losses.mean_squared_error
        Console.WriteSuccess("Compiled. Summary:", self.name)
        self.summary() # Print summary
        self.useLastState = False
        self.ResetLayerStates() # Reset the layer's internal states, just to be sure
        if savedWeightsPath is not None:
            self.TryLoadWeights(savedWeightsPath)

    def call(self, inputs, training=None, mask=None):
        out = inputs
        h = None
        c = None
        for lstmLayer in self.lstmLayers:
            out, h, c = lstmLayer.call(inputs=out, mask=mask, training=training)
        return out

    def CopyWeightsFrom(self, otherLSTMModel: tf.keras.Model):
        self.set_weights(otherLSTMModel.get_weights())
        Console.WriteInfo("Weights copied from %s to %s" % (otherLSTMModel.name, self.name), self.name)

    def TryLoadWeights(self, weightsPath):
        try:
            status = self.load_weights(weightsPath)
            Console.WriteSuccess("Weights loaded from %s" % weightsPath, self.name)
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

    def SetUseLastState(self, newValue: bool):
        if self.useLastState != newValue:
            self.useLastState = newValue
            for lstmLayer in self.lstmLayers:
                lstmLayer.SetUseLastState(newValue)

    def ResetLayerStates(self):
        for lstmLayer in self.lstmLayers:
            lstmLayer.ResetInternalStates()

    def GetStates(self):
        l = []
        for lstmLayer in self.lstmLayers:
            l.append(lstmLayer.GetStates())
        return l

    def SetStates(self, layersStates: list):
        for lstmLayer, hc in zip(self.lstmLayers, layersStates):
            lstmLayer.SetStates(hc)

    def RunPreserved(self, function, **kwargs):
        myStates = self.GetStates() # Save the internal states for later
        self.ResetLayerStates() # Reset the internal states
        result = function(**kwargs) # Execute the function using the keyword arguments
        self.SetStates(myStates) # Load the states back into the LSTM layers
        return result

    def QValues(self, inputs: np.ndarray, useLastState=False, mask=None):
        self.SetUseLastState(useLastState)
        # Call this model to get the output
        return self.call(inputs=inputs, mask=mask)

    def QValuesArgmax(self, inputs: np.ndarray, useLastState=False, mask=None):
        output = self.QValues(inputs, useLastState, mask)
        return tf.argmax(output, axis=2)

    def QValueAt(self, index, inputs: np.ndarray, useLastState=False, mask=None):
        output = self.QValues(inputs, useLastState, mask)
        return output[:,:,index]

    def QValuesNumpy(self, inputs: np.ndarray, useLastState=False, mask=None) -> np.ndarray:
        return tf.keras.backend.eval(self.QValues(inputs, useLastState, mask)) # Evaluate the tensor to get its values

    def QValuesArgmaxNumpy(self, inputs: np.ndarray, useLastState=False, mask=None) -> int:
        return tf.keras.backend.eval(self.QValuesArgmax(inputs, useLastState, mask))

    def QValueAtNumpy(self, index, inputs: np.ndarray, useLastState=False, mask=None) -> np.ndarray:
        return tf.keras.backend.eval(self.QValueAt(index, inputs, useLastState, mask))

    def ApplyGradients(self, gradients):
        self.myOptimizer.apply_gradients(zip(gradients, self.trainable_variables))