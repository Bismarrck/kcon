from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
from itertools import repeat

__author__ = 'Xin Chen'
__email__ = "Bismarrck@me.com"


# Declare the global settings here.
SEED = 235
TF_TYPE = tf.float32
NP_TYPE = np.float32
CUDA_ON = True


def inference(input_tensor, model, **kwargs):
  """

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


def print_activations(t):
  """
  Print the name and shape of the input Tensor.

  Args:
    t: a Tensor.

  """
  dims = ",".join(["%7d" % dim for dim in t.get_shape().as_list()])
  print("%-21s : [%s]" % (t.op.name, dims))


def mbe_conv2d(tensor, n_in, n_out, name="Conv", activate=tf.tanh, verbose=True,
               dropout=False, keep_prob=0.5):
  """ A lazy function to create a `tf.nn.conv2d` Tensor.

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

  if verbose:
    print("-> Inference the MBE-NN-M-FC model ...")
    print("")

  cnk, ck2 = input_tensor.get_shape().as_list()[-2:]

  # Load default settings
  if dims is None:
    dims = [40, 70, 60, 2, 40, 10]
  else:
    assert len(dims) >= 6
  dims = [ck2] + list(dims[:4]) + [cnk] + list(dims[4:])
  activations = {} if activations is None else activations
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
    7. fc1/relu
    8. fc2/relu
    9. output

  Args:
    input_tensor: a Tensor of shape [-1, 1, C(N,k), C(k,2)] as the input layer.
    conv_dims: List[int], the major dims of the conv layers.
    dense_dims: List[int], the size of the dense layers.
    dense_funcs: List, the activation functions of each dense layer. 
      Defaults to ``tf.nn.relu``.
    dropouts: List[int], the indices of the layers to add dropouts.
    conv_keep_prob: a float as the keep probability of conv layers.
    dense_keep_prob: a float as the keep probability of dense layers.
    verbose: a bool.

  Returns:
    y_pred: a Tensor of shape [-1, 1] as the output layer.

  """
  if verbose:
    print("-> Inference the MBE-NN-M-FC model ...")
    print("")

  # Get the height and depth from the input tensor
  cnk, ck2 = input_tensor.get_shape().as_list()[-2:]

  # Switch to the default dimensions if `conv_dims` is not provided.
  if conv_dims is None:
    conv_dims = [40, 70, 60, 2, cnk // 2, cnk // 4]
  else:
    assert len(conv_dims) >= 6

  # Construct the convolutional layers.
  conv = mbe_conv2d(input_tensor, ck2, conv_dims[0], "Conv1")
  num_layers = 1

  for i, major_dim in enumerate(conv_dims[1:]):
    k = i + 1
    num_layers += 1
    nonlinear = tf.nn.tanh if k <= 2 else tf.nn.softplus
    dropout = (k in dropouts)
    scope = "Conv{}".format(num_layers)
    conv = mbe_conv2d(
      conv,
      conv_dims[i],
      conv_dims[i + 1],
      scope,
      activate=nonlinear,
      dropout=dropout,
      keep_prob=conv_keep_prob,
      verbose=verbose
    )

  # Flatten the last convolutional layer so that we can build fully-connected
  # layers.
  dense = tf.contrib.layers.flatten(conv)
  if verbose:
    print_activations(dense)
  num_layers += 1

  # Use the default dense dimensions if it is not provided.
  if dense_dims is None:
    dense_dims = [cnk // 4, cnk // 4]
  else:
    assert len(dense_dims) >= 2

  # Set the default activation functions to ReLu.
  if dense_funcs is None:
    dense_funcs = list(repeat(tf.nn.relu, len(dense_dims)))
  else:
    assert len(dense_funcs) == len(dense_dims)

  # Construct the dense layers
  for i, dim in enumerate(dense_dims):
    name = "Dense%d" % num_layers
    num_layers += 1
    dense = tf.layers.dense(
      dense,
      dim,
      activation=dense_funcs[i],
      name=name
    )
    if verbose:
      print_activations(dense)
    if i + len(conv_dims) in dropouts:
      dense = tf.nn.dropout(dense, keep_prob=dense_keep_prob, seed=SEED)

  # Return the final estimates
  dense = tf.layers.dense(dense, 1, use_bias=False, name="Dense%d" % num_layers)
  if verbose:
    print_activations(dense)
  return dense


def test_mbe_nn_m_fc():
  x_batch = tf.placeholder(tf.float32, [50, 1, 715, 6], name="x_batch")
  keep_prob = tf.placeholder(tf.float32, name="conv_keep_prob")
  dense_keep_prob = tf.placeholder(tf.float32, name="dense_keep_prob")
  _ = inference(
    x_batch,
    "mbe-nn-m-fc",
    dropouts=[2],
    conv_keep_prob=keep_prob,
    dense_keep_prob=dense_keep_prob,
    verbose=True
  )
  nparams = 0
  for var in tf.trainable_variables():
    nparams += np.prod(var.get_shape().as_list(), dtype=int)
  print("")
  print("Total number of parameters: %d" % nparams)


def test_mbe_nn_m():
  x_batch = tf.placeholder(tf.float32, [50, 1, 715, 6], name="x_batch")
  keep_prob = tf.placeholder(tf.float32, name="keep_prob")
  _ = mbe_nn_m(
    x_batch,
    dropouts=[2, 4],
    keep_prob=keep_prob,
    verbose=True
  )
  nparams = 0
  for var in tf.trainable_variables():
    nparams += np.prod(var.get_shape().as_list(), dtype=int)
  print("")
  print("Total number of parameters: %d" % nparams)


if __name__ == "__main__":

  test_mbe_nn_m()
