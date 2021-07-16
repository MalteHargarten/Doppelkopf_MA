import numpy as np
from doppelkopf.utils.Console import Console
#Console.CurrentLevel = Console.LEVEL_OMIT_INFO
from doppelkopf.models.LSTMModel import LSTMModel
from doppelkopf.game.GameState import GameState
from doppelkopf.programs.Program import Program

class TestSuiteLSTM(Program):
    DATATYPE = 'float32'

    def create_dummy_data(self):
        state1 = GameState.Random()
        state2 = GameState.Random()
        state3 = GameState.Random()
        state4 = GameState.Random()
        x1 = state1.Flat().reshape((1, 1, -1)).astype(TestSuiteLSTM.DATATYPE)
        x2 = state2.Flat().reshape((1, 1, -1)).astype(TestSuiteLSTM.DATATYPE)
        x3 = state3.Flat().reshape((1, 1, -1)).astype(TestSuiteLSTM.DATATYPE)
        x4 = state4.Flat().reshape((1, 1, -1)).astype(TestSuiteLSTM.DATATYPE)
        return x1, x2, x3, x4

    def run_predict_individually(self, x1, x2, x3, x4, model: LSTMModel):
        #Console.WriteDebug("# # # # # # # # # # # # # # # my_predict 1 # # # # # # # # # # # # # # #", "main.py")
        output1 = model.QValuesNumpy(x1, useLastState=True)
        Console.WriteDebug("1: " + str(output1), "run_predict_individually()")
        #Console.WriteDebug("# # # # # # # # # # # # # # # my_predict 2 # # # # # # # # # # # # # # #", "main.py")
        output2 = model.QValuesNumpy(x2, useLastState=True)
        Console.WriteDebug("2: " + str(output2), "run_predict_individually()")
        #Console.WriteDebug("# # # # # # # # # # # # # # # my_predict 3 # # # # # # # # # # # # # # #", "main.py")
        output3 = model.QValuesNumpy(x3, useLastState=True)
        Console.WriteDebug("3: " + str(output3), "run_predict_individually()")
        #Console.WriteDebug("# # # # # # # # # # # # # # # my_predict 4 # # # # # # # # # # # # # # #", "main.py")
        output4 = model.QValuesNumpy(x4, useLastState=True)
        Console.WriteDebug("4: " + str(output4), "run_predict_individually()")
        return [output1, output2, output3, output4]

    def test_idempotence_of_individual_inputs(self, x1, x2, x3, x4, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_idempotence_of_individual_inputs'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are always the same across several runs,
            while feeding each state into the network one state at a time.''')
        o1 = []
        o2 = []
        o3 = []
        o4 = []
        for _ in range(2):
            model.ResetLayerStates()
            o = self.run_predict_individually(x1, x2, x3, x4, model)
            o1.append(o[0])
            o2.append(o[1])
            o3.append(o[2])
            o4.append(o[3])        
        if o1[0] == o1[1] and o2[0] == o2[1] and o3[0] == o3[1] and o4[0] == o4[1]:
            Console.WriteSuccess("The results of running the 4 timesteps indidivually after resetting the states are the same", "test_idempotence_of_individual_inputs()")
        else:
            Console.WriteError("The results of running the 4 timesteps indidivually after resetting the states are NOT the same", "test_idempotence_of_individual_inputs()")

    def run_predict_on_sequence(self, sequence, model: LSTMModel):
        Console.WriteDebug("# # # # # # # # # # # # # # # my_predict sequence # # # # # # # # # # # # # # #", "main.py")
        output_sequence = model.QValuesNumpy(sequence, useLastState=False)
        Console.WriteDebug("sequence 1: " + str(output_sequence[0,0]), "run_predict_on_sequence()")
        Console.WriteDebug("sequence 2: " + str(output_sequence[0,1]), "run_predict_on_sequence()")
        Console.WriteDebug("sequence 3: " + str(output_sequence[0,2]), "run_predict_on_sequence()")
        Console.WriteDebug("sequence 4: " + str(output_sequence[0,3]), "run_predict_on_sequence()")
        return output_sequence
    
    def test_idempotence_of_sequence_input(self, sequence, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_idempotence_of_individual_inputs'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are always the same across several runs,
            while feeding all the states to the network as a sequence.''')
        outputs = []
        for _ in range(2):
            model.ResetLayerStates()
            outputs.append(self.run_predict_on_sequence(sequence, model))
        for j in range(4):
            if outputs[0][0,j] == outputs[1][0,j]:
                Console.WriteSuccess("The %dth result of running the 4 timesteps in one sample after resetting the states are the same" % j, "test_idempotence_of_sequence_input()")
            else:
                Console.WriteError("The %dth result of running the 4 timesteps in one sample after resetting the states are NOT the same: %s vs %s" % (j, str(outputs[0][0,j]), str(outputs[1][0,j])), "test_idempotence_of_sequence_input()")

    def test_indenticalness_sequential_vs_individual_inputs(self, x1, x2, x3, x4, sequence, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_indenticalness_sequential_vs_individual_inputs'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are identical, regardless of whether we feed the states
            one at a time or as a sequence.''')
        model.ResetLayerStates()
        o1 = model.QValuesNumpy(x1, useLastState=True).ravel() # Ravel() the result, making it 1D
        o2 = model.QValuesNumpy(x2, useLastState=True).ravel() # Ravel() the result, making it 1D
        o3 = model.QValuesNumpy(x3, useLastState=True).ravel() # Ravel() the result, making it 1D
        o4 = model.QValuesNumpy(x4, useLastState=True).ravel() # Ravel() the result, making it 1D
        model.ResetLayerStates()
        output_sequence = model.QValuesNumpy(sequence, useLastState=False).ravel() # Ravel() the result, making it 1D
        if o1[0] == output_sequence[0]:
            Console.WriteSuccess("First outputs (%s, %s) identical!"% (str(o1[0]), str(output_sequence[0])), "test_indenticalness_sequential_vs_individual_inputs()")
        else:
            Console.WriteError("First output (%s, %s) not identical!" % (str(o1[0]), str(output_sequence[0])), "test_indenticalness_sequential_vs_individual_inputs()")
        if o2[0] == output_sequence[1]:
            Console.WriteSuccess("Second outputs (%s, %s) identical!"% (str(o2[0]), str(output_sequence[1])), "test_indenticalness_sequential_vs_individual_inputs()")
        else:
            Console.WriteError("Second output (%s, %s) not identical!" % (str(o2[0]), str(output_sequence[1])), "test_indenticalness_sequential_vs_individual_inputs()")
        if o3[0] == output_sequence[2]:
            Console.WriteSuccess("Third outputs (%s, %s) identical!"% (str(o3[0]), str(output_sequence[2])), "test_indenticalness_sequential_vs_individual_inputs()")
        else:
            Console.WriteError("Third output (%s, %s) not identical!" % (str(o3[0]), str(output_sequence[2])), "test_indenticalness_sequential_vs_individual_inputs()")
        if o4[0] == output_sequence[3]:
            Console.WriteSuccess("Fourth outputs (%s, %s) identical!"% (str(o4[0]), str(output_sequence[3])), "test_indenticalness_sequential_vs_individual_inputs()")
        else:
            Console.WriteError("Fourth output (%s, %s) not identical!" % (str(o4[0]), str(output_sequence[3])), "test_indenticalness_sequential_vs_individual_inputs()")

    def test_indenticalness_of_normal_vs_custom_predict(self, sequence, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_indenticalness_of_normal_vs_custom_predict'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are identical to the vanilla-prediction
            while feeding the states as a sequence.''')
        custom_prediction = model.QValuesNumpy(sequence, useLastState=False).ravel() # Ravel() the result, making it 1D
        normal_prediction = model.predict(sequence) # , h1, c1, h2, c2, h3, c3 
        normal_prediction = normal_prediction.ravel() # Ravel this to make it 1D
        #Console.WriteSuccess("Normal prediction: %s" % str(normal_prediction), "final_comparison()")
        for i in range(4):
            if normal_prediction[i] == custom_prediction[i]:
                Console.WriteSuccess("%dth result is identical to regular prediction: %s and %s" % (i, str(normal_prediction[i]), str(custom_prediction[i])), "test_indenticalness_of_normal_vs_custom_predict()")
            else:
                Console.WriteSuccess("%dth result is NOT identical to regular prediction: %s and %s" % (i, str(normal_prediction[i]), str(custom_prediction[i])), "test_indenticalness_of_normal_vs_custom_predict()")

    def test_identicalness_of_multiple_sequential_inputs(self, sequence1, sequence2, sequence_full, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_identicalness_of_multiple_sequential_inputs'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are always the same across several runs
            while splitting a single sequence into multiple partial sequences''')
        model.ResetLayerStates()
        out1 = model.QValuesNumpy(sequence1, True).ravel() # Ravel() this to make it 1D
        out2 = model.QValuesNumpy(sequence2, True).ravel() # Ravel() this to make it 1D
        model.ResetLayerStates()
        out_full = model.QValuesNumpy(sequence_full, False).ravel() # Ravel() this to make it 1D
        for i in range(4):
            if i < 2:
                if out_full[i] == out1[i]:
                    Console.WriteSuccess("%dth predictions are identical: %s and %s" % (i, str(out_full[i]), str(out1[i])), "test_identicalness_of_multiple_sequential_inputs()")
                else:
                    Console.WriteError("%dth predictions are NOT identical: %s and %s" % (i, str(out_full[i]), str(out1[i])), "test_identicalness_of_multiple_sequential_inputs()")
            else:
                if out_full[i] == out2[i % 2]:
                    Console.WriteSuccess("%dth predictions are identical: %s and %s" % (i, str(out_full[i]), str(out2[i % 2])), "test_identicalness_of_multiple_sequential_inputs()")
                else:
                    Console.WriteError("%dth predictions are NOT identical: %s and %s" % (i, str(out_full[i]), str(out2[i % 2])), "test_identicalness_of_multiple_sequential_inputs()")

    def test_idempotence_of_preserve(self, x1, x2, x3, x4, sequence, model: LSTMModel):
        Console.WriteInfo('''Beginning 'test_idempotence_of_preserve'.
            The purpose of this function is to establish whether the estimated Q-Values
            of our model are always the same across several runs
            while running a separate sequence in between using 'Preserve' ''')
        # Run the entire sequence normally
        out_normal = model.QValuesNumpy(sequence, useLastState=False) # (1,4,1)
        Console.WriteDebug("out_normal: %s" % (out_normal), "test_idempotence_of_preserve()")
        model.ResetLayerStates()
        out1 = model.QValuesNumpy(x1, useLastState=True).ravel() # (1,)
        out2 = model.QValuesNumpy(x2, useLastState=True).ravel() # (1,)
        out_preserve = model.RunPreserved(model.QValues, inputs=sequence, useLastState=False) # (1,4,1)
        Console.WriteDebug("out_preserve: %s" % (out_preserve), "test_idempotence_of_preserve()")
        out3 = model.QValuesNumpy(x3, useLastState=True).ravel() # (1,)
        out4 = model.QValuesNumpy(x4, useLastState=True).ravel() # (1,)
        Console.WriteDebug("out1: %s" % (out1), "test_idempotence_of_preserve()")
        Console.WriteDebug("out2: %s" % (out2), "test_idempotence_of_preserve()")
        Console.WriteDebug("out3: %s" % (out3), "test_idempotence_of_preserve()")
        Console.WriteDebug("out4: %s" % (out4), "test_idempotence_of_preserve()")
        # # # # # # # # # # # # # # # Compare normal to individual inputs # # # # # # # # # # # # # # # 
        if out1 == out_normal[0,0]:
            Console.WriteSuccess("out1 and out_normal[0,0] are identical: %s and %s" % (out1, out_normal[0,0]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out1 and out_normal[0,0] are NOT identical: %s and %s" % (out1, out_normal[0,0]), "test_idempotence_of_preserve()")
        if out2 == out_normal[0,1]:
            Console.WriteSuccess("out2 and out_normal[0,1] are identical: %s and %s" % (out2, out_normal[0,1]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out2 and out_normal[0,1] are NOT identical: %s and %s" % (out2, out_normal[0,1]), "test_idempotence_of_preserve()")
        if out3 == out_normal[0,2]:
            Console.WriteSuccess("out3 and out_normal[0,2] are identical: %s and %s" % (out3, out_normal[0,2]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out3 and out_normal[0,2] are NOT identical: %s and %s" % (out3, out_normal[0,2]), "test_idempotence_of_preserve()")
        if out4 == out_normal[0,3]:
            Console.WriteSuccess("out4 and out_normal[0,3] are identical: %s and %s" % (out4, out_normal[0,3]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out4 and out_normal[0,3] are NOT identical: %s and %s" % (out4, out_normal[0,3]), "test_idempotence_of_preserve()")
        # # # # # # # # # # # # # # # Compare preserve to individual inputs # # # # # # # # # # # # # # #
        if out1 == out_preserve[0,0]:
            Console.WriteSuccess("out1 and out_preserve[0,0] are identical: %s and %s" % (out1, out_preserve[0,0]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out1 and out_preserve[0,0] are NOT identical: %s and %s" % (out1, out_preserve[0,0]), "test_idempotence_of_preserve()")
        if out2 == out_preserve[0,1]:
            Console.WriteSuccess("out2 and out_preserve[0,1] are identical: %s and %s" % (out2, out_preserve[0,1]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out2 and out_preserve[0,1] are NOT identical: %s and %s" % (out2, out_preserve[0,1]), "test_idempotence_of_preserve()")
        if out3 == out_preserve[0,2]:
            Console.WriteSuccess("out3 and out_preserve[0,2] are identical: %s and %s" % (out3, out_preserve[0,2]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out3 and out_preserve[0,2] are NOT identical: %s and %s" % (out3, out_preserve[0,2]), "test_idempotence_of_preserve()")
        if out4 == out_preserve[0,3]:
            Console.WriteSuccess("out4 and out_preserve[0,3] are identical: %s and %s" % (out4, out_preserve[0,3]), "test_idempotence_of_preserve()")
        else:
            Console.WriteError("out4 and out_preserve[0,3] are NOT identical: %s and %s" % (out4, out_preserve[0,3]), "test_idempotence_of_preserve()")
    
    def onRun(self):
        # # # # # # # # # # # # # # # Create model # # # # # # # # # # # # # # #
        model = LSTMModel.Create(
            name="Test Suite",
            inputType=TestSuiteLSTM.DATATYPE,
            inputSize=GameState.SIZE_STATE,
            hiddenLSTMUnits=[10,5,2],
            outputUnits=1,
            learningRate=0.001)
        # # # # # # # # # # # # # # # Create 4 timesteps # # # # # # # # # # # # # # #
        x1, x2, x3, x4 = self.create_dummy_data()
        # # # # # # # # # # # # # # # Run each timestep one at a time # # # # # # # # # # # # # # #
        self.test_idempotence_of_individual_inputs(x1, x2, x3, x4, model)
        # # # # # # # # # # # # # # # Run all timesteps at the same time # # # # # # # # # # # # # # #
        sequence = np.concatenate((x1, x2, x3, x4)).reshape((1, 4, -1))
        Console.WriteDebug(sequence.shape, "main.py")
        self.test_idempotence_of_sequence_input(sequence, model)
        # # # # # # # # # # # # # # # Comparison of the two methods # # # # # # # # # # # # # # #
        self.test_indenticalness_sequential_vs_individual_inputs(x1, x2, x3, x4, sequence, model)
        sequence1 = np.concatenate((x1, x2)).reshape((1, 2, -1)) # First half of the sequence
        sequence2 = np.concatenate((x3, x4)).reshape((1, 2, -1)) # Second half of the sequence
        self.test_indenticalness_of_normal_vs_custom_predict(sequence, model)
        self.test_identicalness_of_multiple_sequential_inputs(sequence1, sequence2, sequence, model)
        # # # # # # # # # # # # # # # Testing Preserve # # # # # # # # # # # # # # #
        self.test_idempotence_of_preserve(x1, x2, x3, x4, sequence, model)

def main():
    program = TestSuiteLSTM([], [])
    program.Run([])
    
if __name__ == "__main__":
    main()