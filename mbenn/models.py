from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from itertools import repeat
from tensorflow.contrib.layers.python.layers import summaries as summary_lib
from tensorflow.python.framework.ops import GraphKeys


__author__ = 'Xin Chen'
__email__ = "Bismarrck@me.com"


# Declare the global settings here.
SEED = 235
TF_TYPE = tf.float32
NP_TYPE = np.float32
CUDA_ON = True


def get_number_of_trainable_parameters(verbose=False):
  """
  Return the number of trainable parameters in current graph.

  Args:
    verbose: a bool. If True, the number of parameters for each variable will
    also be printed.

  Returns:
    ntotal: the total number of trainable parameters.

  """
  ntotal = 0
  for var in tf.trainable_variables():
    nvar = np.prod(var.get_shape().as_list(), dtype=int)
    if verbose:
      print("{:25s}  {:d}".format(var.name, nvar))
    ntotal += nvar

  print("")
  print("Total number of parameters: %d" % ntotal)


def inference(input_tensor, model, **kwargs):
  """
  The general inference function.

  Args:
    input_tensor: a Tensor of shape [None, 1, C(N,k), C(k,2)].
    model: a string, the name of this model.
    **kwargs: addtional arguments for this model.

  """
  if model.lower() == "mbe-nn-m":

    dims = kwargs.get("dims")
    activations = kwargs.get("activations")
    dropouts = kwargs.get("dropouts")
    keep_prob = kwargs.get("keep_prob", 1.0)
    verbose = kwargs.get("verbose", False)

    return mbe_nn_m(
      input_tensor,
      dims,
      activations,
      dropouts,
      keep_prob,
      verbose,
      return_pred=True
    )

  elif model.lower() == "mbe-nn-m-fc":

    conv_dims = kwargs.get("conv_dims")
    dense_dims = kwargs.get("dense_dims")
    dense_funcs = kwargs.get("dense_funcs")
    dropouts = kwargs.get("dropouts", [])
    conv_keep_prob = kwargs.get("conv_keep_prob", 1.0)
    dense_keep_prob = kwargs.get("dense_keep_prob", 1.0)
    verbose = kwargs.get("verbose", False)

    return mbe_nn_fc(
      input_tensor,
      conv_dims,
      dense_dims,
      dense_funcs,
      dropouts,
      conv_keep_prob,
      dense_keep_prob,
      verbose=verbose
    )

  else:
    raise ValueError("The model %s is not supported!" % model)


def variable_summaries(tensor):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization).

  Args:
    tensor: a Tensor.

  """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(tensor)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tensor, mean))))
      tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(tensor))
    tf.summary.scalar('min', tf.reduce_min(tensor))
    tf.summary.histogram('histogram', tensor)


def print_activations(tensor):
  """
  Print the name and shape of the input Tensor.

  Args:
    tensor: a Tensor.

  """
  dims = ",".join(["{:7d}".format(dim if dim is not None else -1) 
                   for dim in tensor.get_shape().as_list()])
  print("%-21s : [%s]" % (tensor.op.name, dims))


def mbe_conv2d(tensor, n_in, n_out, name="Conv", activate=tf.tanh, verbose=True,
               dropout=False, keep_prob=0.5):
  """
  A lazy function to create a `tf.nn.conv2d` Tensor.

  Args:
    tensor: a Tensor of shape [-1, 1, height, n_in]
    n_in: the number of input channels.
    n_out: the number of output channels.
    name: the name of this layer.
    activate: the activation function, defaults to `tf.tanh`.
    verbose: a bool, if True the layer settings will be printed.
    dropout: a bool, If True a droput tensor will be operated on this layer with
      keep probability `keep_prob`.
    keep_prob: A scalar Tensor with the same type as x. The probability that
      each element is kept.

  Returns:
    activated: a Tensor of activated `tf.nn.conv2d`.

  """
  with tf.name_scope(name):
    with tf.name_scope("filter"):
      kernel = tf.Variable(
        tf.truncated_normal(
          [1, 1, n_in, n_out],
          stddev=0.1,
          seed=SEED,
          dtype=TF_TYPE
        ),
        name="kernel"
      )
      variable_summaries(kernel)
    conv = tf.nn.conv2d(
      tensor,
      kernel,
      [1, 1, 1, 1],
      "SAME",
      use_cudnn_on_gpu=CUDA_ON
    )
    with tf.name_scope("biases"):
      biases = tf.Variable(
        tf.zeros([n_out], dtype=TF_TYPE),
        name="biases"
      )
      variable_summaries(biases)
    bias = tf.nn.bias_add(conv, biases)
    x = activate(bias)
    if verbose:
      print_activations(x)
  if dropout:
    name = "drop{}".format(name)
    drop = tf.nn.dropout(x, keep_prob=keep_prob, name=name, seed=SEED)
    if verbose:
      print_activations(drop)
    return drop
  else:
    return x


def mbe_dense(input_tensor, units, activation=None, use_bias=True,
              dropout=True, keep_prob=0.5, name="Dense", verbose=True):
  """
  A helper function to create a fully-connected layer. The weights will be added
  to the collection `GraphKeys.REGULARIZATION_LOSSES` automatically.

  Args:
    input_tensor: a Tensor of shape [batch, num_in] as the input.
    units: an int, the number of output units.
    activation: the activation function.
    use_bias: a bool. Attach a bias tensor to this layer if True.
    dropout: a bool. If True a droput tensor will be operated on this layer with
      keep probability `keep_prob`.
    keep_prob: A scalar Tensor with the same type as x. The probability that
      each element is kept.
    name: a string, the name of this tensor.
    verbose: a bool.

  Returns:
    dense: a `tf.layers.dense` Tensor.

  """

  batch, nodes = input_tensor.get_shape().as_list()

  with tf.name_scope(name):
    with tf.name_scope("kernel"):
      weights = tf.Variable(
        tf.truncated_normal([nodes, units], stddev=0.1, seed=SEED),
        trainable=True,
        collections=[GraphKeys.REGULARIZATION_LOSSES, 
                     GraphKeys.GLOBAL_VARIABLES],
      )
      variable_summaries(weights)
    dense = tf.matmul(input_tensor, weights)

    if use_bias:
      with tf.name_scope("bias"):
        biases = tf.Variable(
          tf.zeros([units]),
          trainable=True
        )
        variable_summaries(biases)
      dense = tf.add(dense, biases)
    if activation is not None:
      dense = activation(dense)
    if verbose:
      print_activations(dense)

  if dropout:
    name = "drop{}".format(name)
    dense = tf.nn.dropout(dense, keep_prob, seed=SEED, name=name)
    if verbose:
      print_activations(dense)

  return dense


def mbe_nn_m(input_tensor, dims=None, activations=None, dropouts=(),
             keep_prob=1.0, verbose=True, return_pred=True):
  """
  Return the infered MBE-NN-M deep neural network model with 6 convolutional
  layers.

  Args:
    input_tensor: a 4D Tensor as the input layer, [batch, 1, C(N,k), C(k,2)]
    dims: List[int], the major dimensions. The default [40, 60, 70, 2, 40, 10]
      will be used if this is None.
    activations: Dict, the activation functions, { index: func }.
    dropouts: List[int], the indices of the layers to add dropouts.
    keep_prob: a float tensor for dropout layers.
    verbose: a bool. If True, the layer definitions will be printed.
    return_pred: a bool. If False, the last conv layer will be returned.

  Returns:
    res: the estimated result tensor of shape [batch, 1] if `return_pred` is
      True or the last conv2d tensor of shape [batch, 1, dims[3], dims[-1]] will
      be retunred.

  References:
    Alexandrova, A. N. (2016). http://doi.org/10.1021/acs.jctc.6b00994

  """

  # Load default settings
  cnk, ck2 = input_tensor.get_shape().as_list()[-2:]

  if dims is None:
    dims = [40, 70, 60, 2, 40, 10]
  else:
    assert len(dims) >= 6
  dims = [ck2] + list(dims[:4]) + [cnk] + list(dims[4:])
  activations = {} if activations is None else activations
  dropouts = () if dropouts is None else dropouts
  conv = input_tensor

  for i in range(len(dims) - 1):
    if i == 4:
      # Swith the major axis.
      conv = tf.reshape(conv, (-1, 1, dims[4], cnk), name="Switch")
      if verbose:
        print_activations(conv)
    else:
      # Build the first three MBE layers.
      # The shape of the input data tensor is [n, 1, C(N,k), C(k,2)].
      # To fit Fk, the NN connection is localized in the second dimension, and
      # the layer size of the first dimension is kept fixed. The weights and
      # biases of NN connection are shared among different indices of the first
      # dimension, so that the fitted function form of Fk is kept consistent
      # among different k-body terms.
      if i < 4:
        if i == 3:
          activation = activations.get(i, tf.nn.softplus)
        else:
          activation = activations.get(i, tf.nn.tanh)
        drop = bool(i in dropouts)
        name = "Conv{:d}".format(i + 1)
      # Then we build the three mixing layers.
      # The mixing part is used to fit G. Within this part the NN connection is
      # localized in the first dimension, and the size of the second dimension
      # is kept fixed. The parameters of NN connection in this part are shared
      # among different indices of the second dimension.
      else:
        activation = activations.get(i - 1, tf.nn.softplus)
        drop = bool(i - 1 in dropouts)
        name = "Conv{:d}".format(i)
      conv = mbe_conv2d(
        conv,
        dims[i],
        dims[i + 1],
        name=name,
        activate=activation,
        verbose=verbose,
        dropout=drop,
        keep_prob=keep_prob)

  if return_pred:
    # Return the average value of the last conv layer. The average pooling was
    # used in the paper. But `tf.reduce_mean(tf.contrib.layer.flatten)` is more
    # easier to implement.
    flat = tf.contrib.layers.flatten(conv)
    if verbose:
      print_activations(flat)
    y = tf.reduce_mean(
      flat,
      axis=1,
      name="Output",
      keep_dims=True
    )
    if verbose:
      print_activations(y)
    return y
  else:
    # Return the last conv layer so that some dense layers can be appened.
    return conv


def mbe_nn_fc(input_tensor, conv_dims=None, dense_dims=None, dense_funcs=None,
              dropouts=(), conv_keep_prob=1.0, dense_keep_prob=1.0,
              verbose=True):
  """
  Return the infered MBE-NN-M-FC deep neural network model:
    1. conv1/tanh
    2. conv2/tanh
    3. conv3/tanh
    4. conv4/softplus
    5. conv5/softplus
    6. conv6/softplus
    7. dense1/tanh
    8. dense2/tanh
    9. output

  Args:
    input_tensor: a Tensor of shape [-1, 1, C(N,k), C(k,2)] as the input layer.
    conv_dims: List[int], the major dims of the conv layers.
    dense_dims: List[int], the size of the dense layers.
    dense_funcs: List, the activation functions of each dense layer. 
      Defaults to ``tf.nn.tanh``.
    dropouts: List[int], the indices of the layers to add dropouts.
    conv_keep_prob: a float as the keep probability of conv layers.
    dense_keep_prob: a float as the keep probability of dense layers.
    verbose: a bool.

  Returns:
    y_pred: a Tensor of shape [-1, 1] as the output layer.

  """

  cnk, ck2 = input_tensor.get_shape().as_list()[-2:]

  # Switch to the default dimensions if `conv_dims` is not provided.
  if conv_dims is None:
    conv_dims = [40, 70, 60, 2, cnk // 2, cnk // 4]
  else:
    assert len(conv_dims) >= 6

  # Construct the convolutional layers.
  conv = mbe_nn_m(
    input_tensor,
    dims=conv_dims,
    dropouts=dropouts,
    keep_prob=conv_keep_prob,
    verbose=verbose,
    return_pred=False
  )
  num_layers = len(conv_dims)

  # Flatten the last convolutional layer so that we can build fully-connected
  # layers.
  dense = tf.contrib.layers.flatten(conv)
  if verbose:
    print_activations(dense)

  # Load the default dimensions and activation functions if not provided.
  if dense_dims is None:
    dense_dims = list(repeat(conv_dims[-1], 2))
  if dense_funcs is None:
    dense_funcs = list(repeat(tf.nn.tanh, len(dense_dims)))

  # Construct the dense layers
  for i, dim in enumerate(dense_dims):
    drop = bool(num_layers in dropouts)
    name = "Dense{:d}".format(num_layers + 1)
    dense = mbe_dense(dense, dim, activation=dense_funcs[i], use_bias=True,
                      dropout=drop, keep_prob=dense_keep_prob, name=name,
                      verbose=verbose)
    num_layers += 1

  # Return the final estimates
  return tf.reduce_mean(dense, axis=1, name="Output", keep_dims=True)


def test_mbe_nn_m_fc():
  tf.reset_default_graph()

  print("--------------------------")
  print("MBE-NN-M-FC Inference Test")
  print("--------------------------")
  print("")

  x_batch = tf.placeholder(tf.float32, [50, 1, 715, 6], name="x_batch")
  keep_prob = tf.placeholder(tf.float32, name="conv_keep_prob")
  dense_keep_prob = tf.placeholder(tf.float32, name="dense_keep_prob")
  _ = inference(
    x_batch,
    "mbe-nn-m-fc",
    dropouts=[2, 6, 7],
    conv_keep_prob=keep_prob,
    dense_keep_prob=dense_keep_prob,
    verbose=True
  )

  print("")
  print("Calculate the number of parameters: ")
  print("")
  get_number_of_trainable_parameters(verbose=True)

  print("")
  print("The variables that should be regularized:")
  print("")
  regvars = tf.get_collection(GraphKeys.REGULARIZATION_LOSSES)
  for regvar in regvars:
    print(regvar.name)


def test_mbe_nn_m():
  tf.reset_default_graph()

  print("-----------------------")
  print("MBE-NN-M Inference Test")
  print("-----------------------")
  print("")

  x_batch = tf.placeholder(tf.float32, [50, 1, 715, 6], name="x_batch")
  keep_prob = tf.placeholder(tf.float32, name="keep_prob")
  _ = mbe_nn_m(
    x_batch,
    dropouts=[2, 4],
    keep_prob=keep_prob,
    verbose=True
  )

  print("")
  print("Calculate the number of parameters: ")
  print("")
  get_number_of_trainable_parameters(verbose=True)


if __name__ == "__main__":

  test_mbe_nn_m()
  test_mbe_nn_m_fc()
