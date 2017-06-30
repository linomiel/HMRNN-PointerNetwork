import numpy as np
import tensorflow as tf

from keras import initializations
from keras.layers.recurrent import time_distributed_dense
from keras.activations import tanh, softmax
from keras.layers import LSTM
from keras.engine import InputSpec
import keras.backend as K

class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape # this is not used by LSTM. We can handle it at will.
        self.input_length = []  
        super(PointerLSTM, self).__init__(*args, **kwargs) # units = 2*hidden_shape[2]?

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        init = initializations.get('orthogonal')
        #dimension problem!!!!
        self.W1 = init((self.hidden_shape, 1))
        self.W2 = init((self.hidden_shape, 1))
        self.W3 = init((self.hidden_shape, 1))
        self.W4 = init((self.hidden_shape, 1))
        self.W5 = init((self.hidden_shape, 1))
        self.vt = init((hidden_shape, 1))
        self.trainable_weights += [self.W1, self.W2, self.W3, self.W4, self.W5, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape # which shape? [sample, time, vector]. Why.shape?    
        en_seq1, en_seq2 = x 
        x1_input = en_seq1[:, input_shape[1]-1, :] 
        x2_input = en_seq2[:, input_shape[1]-1, :]
        x1_input = K.repeat(x1_input, input_shape[1])
        x2_input = K.repeat(x2_input, input_shape[1])
        initial_states1 = self.get_initial_states(x1_input)
        initial_states2 = self.get_initial_states(x2_input)
        initial_states =  initial_states1 + initial_states2
        constants1 = super(PointerLSTM, self).get_constants(x1_input)
        constants2 = super(PointerLSTM, self).get_constants(x2_input)
        constants = constants1 + constants2
        constants.append(en_seq1)
        constants.append(en_seq2)
        preprocessed_input1 = self.preprocess_input(x1_input)
        preprocessed_input2 = self.preprocess_input(x2_input)
        preprocessed_input = tf.concat([preprocessed_input1, preprocessed_input2], 2, name ='preprocessed_input')
        #call of the step function. The arguments are fixed by K.rnn. Need to fit the 2 layers input and states
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq2 = states[-1]
        en_seq1 = states[-2]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-2]) # all states, but the en_seq

        # v*tanh(W1*e1+W2*d)
        # v = W3*c
        # c = sum(e2*b)
        # b = vt*tanh(W4*e2+W5*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij_1 = time_distributed_dense(en_seq1, self.W1, output_dim=1)
        Dij_1 = time_distributed_dense(dec_seq, self.W2, output_dim=1)
        Eij_2 = time_distributed_dense(en_seq2, self.W4, output_dim=1)
        Dij_2 = time_distributed_dense(dec_seq, self.W5, output_dim=1)
        B = self.vt * tanh(Eij_2 + Dij_2)
        B = K.squeeze(B, 2)
        beta = softmax(B)
        c = K.transpose(beta * K.transpose(en_seq2))
        c = K.sum(c, 0)
        v = K.dot(self.W3, c)
        U = v * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0][0], input_shape[0][1], input_shape[0][1])#depend on the input form.

my_pointer = PointerLSTM(hidden_shape=50,units=100)#hidden_shape 
first_layer_input = tf.placeholder(tf.float32, [1,None,50], name = 'first_layer_input')
second_layer_input = tf.placeholder(tf.float32, [1,None,50], name = 'second_layer_input')
y = my_pointer(first_layer_input, second_layer_input)