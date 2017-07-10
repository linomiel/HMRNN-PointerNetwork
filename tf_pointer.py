import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib import seq2seq

from decoder import dynamic_rnn_decoder
from decoder import simple_decoder_fn_train

from functools import partial

def _affine(args, output_size, bias=True, scope=None, init_W=None):
  # Calculate the total size of arguments on dimension 1
  total_arg_size = 0
  shapes = [arg.get_shape() for arg in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value
  # Set data type
  dtype = args[0].dtype
  # Compute linear part
  _scope = tf.get_variable_scope()
  with tf.variable_scope(_scope) as outer_scope:
    with tf.variable_scope(scope) as inner_scope:
      if init_W is not None:
        W = tf.get_variable('W', initializer=init_W, dtype=dtype)
      else:
        W = tf.get_variable('W', [total_arg_size, output_size], dtype=dtype)
      tf.add_to_collection('weights', W)
      tf.add_to_collection('vars', W)
      if len(args) == 1:
        logits = math_ops.matmul(args[0], W)
      else:
        logits = math_ops.matmul(array_ops.concat(args, 1), W)
      if not bias:
        return logits
      b = tf.get_variable('b', [output_size], dtype=dtype,
        initializer=init_ops.constant_initializer(0.0, dtype=dtype))
      tf.add_to_collection('vars', b)
  return nn_ops.bias_add(logits, b)

def _lstm_gates(logits, num_splits=4, axis=1, activation=tanh, forget_bias=0.0):
    """Split logits into input, forget, output and candidate
    logits and apply appropriate activation functions.
    _input: input gates, _forget: forget gates,
    _output: output gates,  _cell: cell candidates
    """
    _input, _forget, _output, _cell = \
        array_ops.split(value=logits, num_or_size_splits=num_splits, axis=axis)
    _input = sigmoid(_input)
    _forget = sigmoid(_forget + forget_bias)
    _output = sigmoid(_output)
    _cell = activation(_cell)
    return _input, _forget, _output, _cell

# class LSTMCell(RNNCell):
#   """LSTM recurrent network cell.
#   """
#   def __init__(self, num_units, forget_bias=0.0, activation=tanh):
#     """Initialize the LSTM cell.
#     Args:
#       num_units: int, the number of units in the LSTM cell.
#       forget_bias: float, The bias added to forget gates (see above).
#       activation: Activation function of the inner states.
#     """
#     self.num_units = num_units
#     self.forget_bias = forget_bias
#     self.activation = activation

#   @property
#   def state_size(self):
#     return LSTMStateTuple(self.num_units, self.num_units)

#   @property
#   def output_size(self):
#     return self.num_units

#   def __call__(self, inputs, state, scope='lstm'):
#     """Long Short-Term Memory (LSTM) cell.
#     """
#     # Parameters of gates are concatenated into one multiply for efficiency.
#     c, h = state
#     logits = _affine([inputs, h], 4 * self.num_units, scope=scope)
#     i, f, o, j = _lstm_gates(logits, forget_bias=self.forget_bias)
#     # Update the states
#     new_c = c * f + i * j
#     new_h = o * self.activation(new_c)
#     # Update the returns
#     new_state = LSTMStateTuple(new_c, new_h)
#     return new_h, new_state




class AttentionModule(object):
  """Implementation of LSTM module"""

  def __init__(self, units, enc_units_1, enc_units_2, hidden_dim, max_length = 1000, num_glimpse = 1, initializer = None):
    #max_length: the max number of decoder iteration
    self.units = units
    self.enc_units_1 = enc_units_1 
    self.enc_units_2 = enc_units_2 
    self.hidden_dim = hidden_dim # in all generality, 2 different could be used. One for W1, W2 and W3, and a different for v W4 and W5 (cf dimensionality equations)
    self.max_length = max_length
    self.num_glimpse = num_glimpse
    self.initializer = initializer
  
  # def attention(enc1, enc2, dec, with_softmax, scope="attention"):
  #   #takes as imput the batches of sequences of encoder and the last decoder state.
  #   #generates a probability 1D vector.
  #     with tf.variable_scope(scope):
  #       W_1 = tf.get_variable(
  #           "W_1", [hidden_dim, enc_units_1], initializer=initializer)
  #       W_2 = tf.get_variable(
  #           "W_2", [hidden_dim, units], initializer=initializer)
  #       W_3 = tf.get_variable(
  #           "W_3", [hidden_dim, enc_units_2], initializer=initializer)
  #       W_4 = tf.get_variable(
  #           "W_4", [hidden_dim, enc_units_2], initializer=initializer)
  #       W_5 = tf.get_variable(
  #           "W_5", [hidden_dim, units], initializer=initializer)
  #       v_2 = tf.get_variable(
  #           "v", [hidden_dim], initializer=initializer)
  #       #collects the shapes
  #       enc1_shape = enc1.shape
  #       enc2_shape = enc2.shape
  #       dec_shape = dec.shape
  #       #collapses the batch and time dimensions together
  #       enc1 = tf.reshape(enc1, [-1, enc1_shape[2]])
  #       enc2 = tf.reshape(enc2, [-1, enc2_shape[2]])
  #       #dec is already 1D
  #       #applies the matmul op to each input
  #       Enc_1 = tf.scan(lambda x: tf.matmul(W_1,x), enc1)
  #       Dec_1 = tf.matmul(W_2,dec)
  #       Enc_2 = tf.scan(lambda x: tf.matmul(W_4,x), enc2)
  #       Dec_2 = tf.matmul(W_5,dec)
  #       # transpose of v_2?
  #       B = tf.scan(lambda x: tf.matmul(v_2,x), Enc_2 + Dec_2)
  #       b = tf.nn.softmax(B)
  #       c = tf.reduce_sum(Dec_2,b)
  #       #note that v_1 has shape [None, hidden_dim] != v_2.shape
  #       v_1 = tf.scan(lambda x: tf.matmul(W_3,x), c)
  #       scores = tf.scan(lambda x, y: tf.matmul(x, y,), (v_1,Enc_1 + Dec_1))
  #       #makes it a 1D [None] tensor
  #       scores = tf.squeeze(scores)

  #       if with_softmax:
  #         return tf.nn.softmax(scores)
  #       else:
  #         return scores

  # def glimpse(enc1, enc2, dec, scope="glimpse"):
  #     p = attention(enc1, enc2, dec, with_softmax=True, scope=scope)
  #     alignments = tf.expand_dims(p, 2)
  #     return tf.reduce_sum(alignments * ref, [1])

  # def output_fn(enc1, enc2, dec, num_glimpse):
  #   if query is None:
  #     return tf.zeros([max_length], tf.float32) # only used for shape inference
  #   else:
  #     for idx in range(num_glimpse):
  #       query = glimpse(enc1, enc2, dec, "glimpse_{}".format(idx))
  #     return attention(enc1, enc2, dec, with_softmax=False, scope="attention")

  # def input_fn(sampled_idx):
  #   return tf.stop_gradient(
  #       tf.gather_nd(enc_outputs, index_matrix_to_pairs(sampled_idx)))

  # def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
  #   cell_output = output_fn(self.enc1, self.enc2, cell_output, num_glimpse)
  #   if cell_state is None:
  #     cell_state = enc_final_states
  #     next_input = cell_input
  #     done = tf.zeros([batch_size,], dtype=tf.bool)
  #   else:
  #     sampled_idx = tf.cast(tf.argmax(cell_output, 1), tf.int32)
  #     next_input = input_fn(sampled_idx)
  #     done = tf.equal(sampled_idx, end_of_sequence_id)

  #   maximum_length = tf.convert_to_tensor(max_length, tf.int32)
  #   done = tf.cond(tf.greater(time, maximum_length),
  #     lambda: tf.ones([batch_size,], dtype=tf.bool),
  #     lambda: done)
  #   return (done, cell_state, next_input, cell_output, context_state)

  def __call__(self, inputs, enc1, enc2 , is_train = False, batch_size = 1):
    rnn_cell = LSTMCell(self.units)

    self.enc1 = enc1
    self.enc2 = enc2

    seq_length = inputs.shape[2]
    if is_train:
      decoder_fn = simple_decoder_fn_train(enc_final_states)
    else:
      def attention(enc1, enc2, dec, with_softmax, scope="attention"):
      #takes as imput the batches of sequences of encoder and the last decoder state.
      #generates a probability 1D vector.
        with tf.variable_scope(scope):
          W_1 = tf.get_variable(
              "W_1", [self.hidden_dim, self.enc_units_1], initializer=self.initializer)
          W_2 = tf.get_variable(
              "W_2", [self.hidden_dim, self.units], initializer=self.initializer)
          W_3 = tf.get_variable(
              "W_3", [self.hidden_dim, self.enc_units_2], initializer=self.initializer)
          W_4 = tf.get_variable(
              "W_4", [self.hidden_dim, self.enc_units_2], initializer=self.initializer)
          W_5 = tf.get_variable(
              "W_5", [self.hidden_dim, self.units], initializer=self.initializer)
          v_2 = tf.get_variable(
              "v_2", [1,self.hidden_dim], initializer=self.initializer)
          #collapses the batch and time dimensions together
          enc1 = tf.reshape(enc1, [-1, self.enc_units_1])
          enc2 = tf.reshape(enc2, [-1, self.enc_units_2])
          #dec is already 2D [batch, vector]
          #applies the matmul op to each input
          apply_mult = lambda matrix, init, x : tf.squeeze(tf.matmul(matrix, tf.expand_dims(x,1)))
          Enc_1 = tf.scan(partial(apply_mult, W_1) , enc1, initializer = tf.zeros(self.hidden_dim))
          Dec_1 = tf.scan(partial(apply_mult, W_2) , dec, initializer = tf.zeros(self.hidden_dim))
          Enc_2 = tf.scan(partial(apply_mult, W_4), enc2, initializer = tf.zeros(self.hidden_dim))
          Dec_2 = tf.scan(partial(apply_mult, W_5) , dec, initializer = tf.zeros(self.hidden_dim))
          # add is supposed to support broadcasting.
          B = tf.scan(partial(apply_mult, v_2), tf.add(Enc_2, Dec_2), initializer = tf.zeros(())  )
          b = tf.nn.softmax(B)
          
          apply_b = lambda init, (scalar, vec) : tf.scalar_mul(scalar,vec)
          weighted_dec2 = tf.scan(apply_b, (b,Dec_2), initializer = tf.zeros_like(Dec_2[0]))
          c = tf.reduce_sum(weighted_dec2, axis = 0, name = 'c')
          #note that v_1 has shape [None, hidden_dim] != v_2.shape
          v_1 = tf.scan(lambda init, x: tf.matmul(W_3,x), c)
          scores = tf.scan(lambda init, (x, y): tf.matmul(x, y,), (v_1,Enc_1 + Dec_1))
          #makes it a 1D [None] tensor
          scores = tf.squeeze(scores)

          if with_softmax:
            return tf.nn.softmax(scores)
          else:
            return scores

    def glimpse(enc1, enc2, dec, scope="glimpse"):
        p = attention(enc1, enc2, dec, with_softmax=True, scope=scope)
        alignments = tf.expand_dims(p, 2)
        return tf.reduce_sum(alignments * ref, [1])

    def output_fn(enc1, enc2, dec, num_glimpse):
      if dec is None:
        return tf.zeros([self.max_length], tf.float32) # only used for shape inference
      else:
        for idx in range(num_glimpse):
          dec = glimpse(enc1, enc2, dec, "glimpse_{}".format(idx))
        return attention(enc1, enc2, dec, with_softmax=False, scope="attention")

    def input_fn(sampled_idx):
      return tf.stop_gradient(
          tf.gather_nd(enc_outputs, index_matrix_to_pairs(sampled_idx)))

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
      cell_output = output_fn(self.enc1, self.enc2, cell_output, self.num_glimpse)
      if cell_state is None:
        #cell_state = enc_final_states
        select_last = lambda init, x: x[-1]
        final1 = tf.scan(select_last,enc1, initializer= tf.zeros([self.enc_units_1]))
        final2 = tf.scan(select_last,enc2, initializer= tf.zeros([self.enc_units_2]) )
        final_enc_state =  tf.concat([final1,final2],1, name = 'final_enc_state')
        W_convert = tf.get_variable("W_convert", [2*self.units, self.enc_units_1+self.enc_units_2], initializer = self.initializer)#In order to keep an independent unit size. A good idea?
        cell_state = tf.scan(lambda init, x: tf.squeeze(tf.matmul(W_convert,tf.expand_dims(x,1))), final_enc_state, initializer = tf.zeros([2*self.units]), name = 'decorder_fn_cell_state') #expand_dims for shape coherance.
        c,h = tf.split(cell_state,2, axis= 1)
        cell_state = LSTMStateTuple(c,h)
        next_input = cell_input
        done = tf.zeros([batch_size,], dtype=tf.bool)
      else:
        sampled_idx = tf.cast(tf.argmax(cell_output, 1), tf.int32)
        next_input = input_fn(sampled_idx)
        done = tf.equal(sampled_idx, end_of_sequence_id)

      maximum_length = tf.convert_to_tensor(self.max_length, tf.int32)
      done = tf.cond(tf.greater(time, maximum_length),
        lambda: tf.ones([batch_size,], dtype=tf.bool),
        lambda: done)
      return (done, cell_state, next_input, cell_output, context_state)

    outputs, final_state, final_context_state = \
        dynamic_rnn_decoder(LSTMCell(num_units = self.units), decoder_fn, inputs=inputs, sequence_length=seq_length)

    return outputs, last_state # we don't need final_context_state (i think)

##compiles?
my_model = AttentionModule(50,100,100,50)
e1 = tf.placeholder(tf.float32, shape = [None,None,100], name = 'e1')
e2 = tf.placeholder(tf.float32, shape = [None,None,100], name = 'e2')
x = tf.placeholder(tf.float32, shape = [None,None,50], name = 'x')
y = my_model(x,e1,e2)