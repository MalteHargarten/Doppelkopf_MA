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
from doppelkopf.utils.Console import Console

class LSTMLayer(tf.keras.layers.LSTM):
    def __init__(self, units: int, name: str): # incoming_tensor
        super(LSTMLayer, self).__init__(units=units, name=name, return_sequences=True, return_state=True)
        self.outputTensor = None
        self.lastHiddenState = None
        self.lastCellState = None
        self.useLastState = False
        self.supports_masking = True # To support Mask propagation (Opting-in)

    def SetUseLastState(self, newValue: bool):
        if self.useLastState != newValue:
            self.useLastState = newValue

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if self.useLastState:
            if self.lastHiddenState is not None and self.lastCellState is not None:
                initial_state = [self.lastHiddenState, self.lastCellState]
        self.outputTensor, self.lastHiddenState, self.lastCellState = super(LSTMLayer, self).call(inputs=inputs, mask=mask, training=training, initial_state=initial_state)
        return self.outputTensor, self.lastHiddenState, self.lastCellState

    def ResetInternalStates(self):
        self.lastHiddenState = None
        self.lastCellState = None

    def SetStates(self, states: tuple):
        hs, cs = states # Unpack tuple
        self.lastHiddenState = hs
        self.lastCellState = cs

    def GetStates(self):
        return (self.lastHiddenState, self.lastCellState) # Pack as tuple
