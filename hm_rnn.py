import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.framework import ops

#high level class, to be called for the model construction
class HMLSTMModule(object):
  """Implementation of Hierarchical Multiscale LSTM module"""

  def __init__(self, num_units, copy_last_layer):
    self.num_units = num_units
    self.copy_last_layer = copy_last_layer

  def __call__(self, inputs, init_state, init_boundary, scope=None):
    state_list = tf.unstack(init_state, axis=0)
    #build the hm_lstm state (mirror the stacking process?)
    rnn_tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(
         state_list[idx][0], state_list[idx][1])
         for idx in range(len(state_list))] +
        [init_boundary[0]])
    # Define an LSTM cell with Tensorflow (class defined below)
    rnn_cell = HMLSTMCell(self.num_units, copy_last_layer=self.copy_last_layer)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs,
                                        initial_state=rnn_tuple_state,
                                        time_major=True)
    # Stack last boundaries of vectors to a matrix
    last_boundary = states[-1]
    # Stack last states of matrices to a 4-D tensor
    last_state = tf.stack([tf.stack([state[0], state[1]], axis=0)
                           for state in states[:2]], axis=0)
    return outputs, last_state, last_boundary

##initialiazation functions
def custom_init(nin, nout=None, scale=0.01, orthogonal=True):
  if nout is None:
    nout = nin
  if nout == nin and orthogonal:
    x = orthogonal_init(nin)
  else:
    x = glorot_init(nin, nout)
  return x

def glorot_init(nin, nout=None, uniform=True):
  if nout is None:
    nout = nin
  if uniform:
    scale = np.sqrt(6.0 / (nin + nout))
    x = uniform_init(nin, nout, scale)
  else:
    scale = np.sqrt(3.0 / (nin + nout))
    x = normal_init(nin, nout, scale)
  return x

def uniform_init(nin, nout=None, scale=0.01):
  x = np.random.uniform(size=(nin, nout), low=-scale, high=scale)
  return x.astype(np.float32)

def normal_init(nin, nout=None, scale=0.01):
  x = scale * np.random.normal(loc=0.0, scale=1.0, size=(nin, nout))
  return x.astype(np.float32)

def orthogonal_init(nin):
  x = np.random.normal(0.0, 1.0, (nin, nin))
  u, _, v = np.linalg.svd(x, full_matrices=False)
  return u.astype(np.float32)

#create the gate mecanism.
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

##Activation functions for the straight through estimator.

# Tensorflow Activation
def hard_sigmoid(x, scale=1.):
  return tf.clip_by_value((scale * x + 1.) / 2., clip_value_min=0,
                          clip_value_max=1)


# Tensorflow Op
def binary_round(x):
  """Rounds a tensor whose values are in [0, 1] to a tensor
  with values in {0, 1}, using the straight through estimator
  to approximate the gradient.
  """
  g = tf.get_default_graph()
  with ops.name_scope("binary_round") as name:
    with g.gradient_override_map({"Round": "Identity"}):
      return tf.round(x, name=name)


def binary_sigmoid(x, slope_tensor=None):
  """Straight through hard sigmoid.
  Hard sigmoid followed by the step function.
  """
  if slope_tensor is None:
    slope_tensor = tf.constant(1.0)
  p = hard_sigmoid(x, slope_tensor)
  return binary_round(p)

##Core class of the HMLSTM model.
class HMLSTMCell(RNNCell):
  """Hierarchical Multiscale LSTM recurrent network cell.
  Three-layered, use binary straight-through.
  """
  def __init__(self, num_units, forget_bias=0.0, copy_last_layer=False,
               activation=tanh):
    """Initialize the Hierarchical Multiscale LSTM cell.
    Args:
      num_units: int, the number of units in the HM-LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      to forget gate connection.
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.activation = activation
    self.copy_last_layer = copy_last_layer

  @property
  def state_size(self): #strange. Same size for each layer?
    return (LSTMStateTuple(self.num_units, self.num_units),
            LSTMStateTuple(self.num_units, self.num_units))

  @property
  def output_size(self):
    return (self.num_units, self.num_units, 1)

  def __call__(self, inputs, state, scope='hm_lstm'):
    """Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) cell.
    State is a tuple of the first-layer c1 and h1, the second-layer
    c2 and h2, the third-layer c3 and h3, the first-layer boundary
    detector z1 and the second-layer boundary detector z2 at previous
    time step.
    """
    # States from previous time step
    (c1, h1), (c2, h2), z = state
    # First RNN
    logits1 = self.f_hmlstm(inputs, h1, z * h2, scope='first_rnn')
    # Split and apply activation
    i1, f1, o1, j1 = _lstm_gates(logits1[:, :-1])
    # Update the states
    is_update = 1. - z
    # if is_flush: z1 == 1 only i1 * j1 is passed
    new_c1 = is_update * f1 * c1 + i1 * j1
    new_h1 = o1 * tanh(new_c1)
    # Straight-through estimation
    new_z = binary_sigmoid(logits1[:, -1])
    new_z = tf.expand_dims(new_z, axis=1)
    # Second RNN
    logits2 = self.f_hmlstm(new_z * new_h1, h2, scope='second_rnn',
                            is_last_layer=True)
    # Split and apply activation
    i2, f2, o2, j2 = _lstm_gates(logits2)
    # Update the states
    new_c2 = f2 * c2 + i2 * j2
    new_c2 = new_z * new_c2 + (1. - new_z) * c2
    if self.copy_last_layer:
      new_h2 = new_z * o2 * tanh(new_c2) + (1. - new_z) * h2
    else:
      new_h2 = o2 * tanh(new_c2)
    # Update the returns
    new_state1 = LSTMStateTuple(new_c1, new_h1)
    new_state2 = LSTMStateTuple(new_c2, new_h2)
    new_state = (new_state1, new_state2, new_z)
    return (new_h1, new_h2, new_z), new_state

  def f_hmlstm(self, h_below, h_before, h_above=None, scope=None,
               is_last_layer=False):
    if is_last_layer:
      W = self.custom_block_initializer(h_below.shape[1], self.num_units,
                                        is_z=False)
      logits = _affine([h_below, h_before], 4 * self.num_units,
                       scope=scope, init_W=W)
      return logits
    W = self.custom_block_initializer(h_below.shape[1], self.num_units,
                                      is_topdown=True)
    logits = _affine([h_below, h_before, h_above], 4 * self.num_units + 1,
                     scope=scope, init_W=W)
    return logits

  def custom_recurrent_initializer(self, num_units, is_z=True):
    x = orthogonal_init(num_units)
    U = np.concatenate([x] * 4, axis=1)
    if is_z:
      z = glorot_init(num_units, 1)
      U = np.concatenate([U, z], axis=1)
    return U

  def custom_block_initializer(self, num_input_units, num_units, is_z=True,
                               is_topdown=False):
    """Custom weight initalizer for HM-LSTM using numpy arrays.
    Block initialization for RNNs.
    Args:
      num_input_units: number of input units
      num_units: number of hidden units, assume equal for every layer
    """
    try:
      num_input_units = num_input_units.value
    except:
      pass
    x = custom_init(num_input_units, num_units)
    W = np.concatenate([x] * 4, axis=1)
    if is_z:
      z = glorot_init(num_input_units, 1)
      W = np.concatenate([W, z], axis=1)
    U = self.custom_recurrent_initializer(num_units, is_z)
    W = np.vstack([W, U])
    if is_topdown:
      W = np.vstack([W, U])
    return W


## compiles?
my_rnn = HMLSTMModule(100, False)
x = tf.placeholder(tf.float32, shape=(None,None,100), name = 'x')
state = tf.placeholder(tf.float32, [2, 2, None, 100], name = 'state')
boundary = tf.placeholder(tf.float32, [1, None, 1], name = 'boundary')
y = my_rnn(x, state, boundary)
print(y[0][0].shape)
print(y[0][1].shape)
print(y[1].shape)


##runs?
sess = tf.Session()
sess.run(tf.global_variables_initializer())#for some reason, the bias in _affine is not well initialized. So this is needed. (bug or feature?)
print(sess.run(y, {x : np.random.random((1,5,100)), state : np.random.random((2,2,5,100)), boundary : np.random.random((1,5,1))}))