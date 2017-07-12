import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq


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


class AttentionModule(object):
  def __init__(self, units, hidden_dim, enc_units_1, enc_units_2, f_bias = 0.0, activation = tanh, num_glimpse = 1, initializer = None):
    self.units = units
    self.hidden_dim = hidden_dim
    self.enc_units_1 = enc_units_1
    self.enc_units_2 = enc_units_2
    self.forget_bias = forget_bias
    self.activation = activation
    self.num_glimpse = num_glimpse
    self.initializer = initializer

  def __call__(self, inputs, state):
    #The first state has to be handled here.
    #How to differentiate between train and test time, while sharing the cell? 
    pass


def input_fn(sampled_idx, enc1):
  return tf.stop_gradient(
    tf.gather_nd(enc1, index_matrix_to_pairs(sampled_idx)))


class AttentionCell(RNNCell):

  def __init__(self, units, hidden_dim, enc_units_1, enc_units_2, f_bias = 0.0, activation = tanh, num_glimpse = 1, initializer = None):
    self.units = units # so far, units = enc_units_1 is necessary. cf call functions
    self.hidden_dim = hidden_dim
    self.enc_units_1 = enc_units_1
    self.enc_units_2 = enc_units_2
    self.forget_bias = forget_bias
    self.activation = activation
    self.num_glimpse = num_glimpse
    self.initializer = initializer

  @property
  def state_size(self):
    return LSTMStateTuple(self.units, self.units)

  @property
  def output_size(self):
    return self.units

  def attention(enc1, enc2, dec, with_softmax, scope="attention"):
  #enc1.shape = [batch_size, length_of_encoder_sequence, enc_units_1]
  #enc2.shape = [batch_size, length_of_encoder_sequence, enc_units_2]
  #dec.shape = [batch_size, units]
  #generates a 2D vector of probabilities. Shape : [batch_size, length_of_encoder_sequence].
    with tf.variable_scope(scope):
      W_1 = tf.get_variable(
          "W_1", [self.enc_units_1, self.hidden_dim], initializer=self.initializer)
      W_2 = tf.get_variable(
          "W_2", [self.units, self.hidden_dim], initializer=self.initializer)
      W_3 = tf.get_variable(
          "W_3", [self.enc_units_2, self.hidden_dim], initializer=self.initializer)
      W_4 = tf.get_variable(
          "W_4", [self.enc_units_2, self.hidden_dim], initializer=self.initializer)
      W_5 = tf.get_variable(
          "W_5", [self.units, self.hidden_dim], initializer=self.initializer)
      v_2 = tf.get_variable(
          "v_2", [self.hidden_dim], initializer=self.initializer)

      Enc_1 = tf.nn.conv1d(enc1, W_1, 1, "VALID", name="Enc_1")
      #Enc_1.shape = [batch, length_of_encoder_sequence, hidden_dim]
      Dec_1 =  tf.expand_dims(tf.matmul(dec, W_2, name="Dec_1"), 1)
      #Dec_1.shape = [batch, 1, hidden_dim]
      Enc_2 = tf.nn.conv1d(enc1, W_4, 1, "VALID", name="Enc_2")
      Dec_2 = tf.expand_dims(tf.matmul(dec, W_5, name="Dec_2"), 1)
      B = tf.reduce_sum(v_2 * tf.tanh(Enc_2 + Dec_2), [-1])
      #B.shape = [batch, length_of_encoder_sequence]
      b = tf.expand_dims(tf.nn.softmax(B), 2)
      #b.shape = [batch, length_of_encoder_sequence, 1]
      c = tf.reduce_sum(enc2*b, 1)
      #c.shape = [batch, enc_units_2]
      v_1 = tf.expand_dims(tf.matmul(c, W_3, name = 'v_1'),1)
      #v_1.shape = [batch, 1, hidden_dim]
      scores = tf.reduce_sum(v_1 * tf.tanh(Enc_1 + Dec_1), [-1])

      if with_softmax:
        return tf.nn.softmax(scores)
      else:
        return scores

  def glimpse(enc1, enc2, dec, scope="glimpse"):
    p = attention(enc1, enc2, dec, with_softmax=True)#, scope=scope
    alignments = tf.expand_dims(p, 2)
    return tf.reduce_sum(alignments * enc1, [1])

  #if dec is none to fix
  def output_fn(enc1, enc2, dec, num_glimpse):
    # if dec is None:
    #   return tf.zeros([self.max_length], tf.float32) # only used for shape inference
    # else:
      for idx in range(num_glimpse):
        dec = glimpse(enc1, enc2, dec, "glimpse_{}".format(idx))
      return attention(enc1, enc2, dec, with_softmax=False, scope="attention")
 

  def __call__(self, inputs, state, scope ='attention_cell'):
    #inputs is a tuple: (enc1, enc2, main_input) with
      # enc1.shape = [batch_size*time*enc_units_1]
      # enc1.shape = [batch_size*time*enc_units_1]
      # main_input = [batch_size*units]
    #state is a LSTMStateTuple (c,h) where c.shape = h.shape = [batch_size*units]
      # The first state has to be initialized out of the cell!
    c, h = state
    enc1, enc2, main_input = inputs

    #if this is not the first call, focus on the correct timesteps of enc1
    is_zero = tf.equal(main_input, tf.zeros_like(main_input))
    default_input = tf.scan( lambda init, x : init & x, is_zero, initializer = True)
    if not default_input:
      #turns the probabilities into pointers.
      sampled_idx = tf.cast(tf.argmax(main_input, 1), tf.int32)
      pointers = input_fn(sampled_idx)
      #use the pointers to select the correct enc1 output.
      main_input = tf.scan(lambda init, (x,p) : x[p], (enc1,pointers), initializer = tf.zeros(self.enc_units_1))
      #main_input.shape=[batch_size, enc_units_1] This is why enc_units_1 should be units.
    logits = _affine([main_input, h], 4 * self.units)#, scope=scope
    i, f, o, j = _lstm_gates(logits, forget_bias=self.forget_bias)
    new_c = c * f + i * j
    #activation required?
    new_h = o * self.activation(new_c)
    cell_output = output_fn(enc1,enc2,new_h,self.num_glimpse)
    

    new_state = LSTMStateTuple(new_c, new_h)
    return cell_output, new_state