from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

__author__ = 'Xin Chen'
__email__ = "Bismarrck@me.com"


# Declare the global settings here.
SEED = 235
TF_TYPE = tf.float32
NP_TYPE = np.float32
CUDA_ON = True


def inference(input_tensor, keep_prob, model, **kwargs):
  """

  Args:
    input_tensor: a Tensor of shape [None, 1, C(N,k), C(k,2)].
    keep_prob: the dropout keep probility.
    model: a string, the name of this model.
    **kwargs: addtional arguments for this model.

  """
  if model.lower() == "mbe-nn-m-6":
    if "dims" in kwargs:
      dims = kwargs["dims"]
    else:
      dims = None
    return mbe_nn_m_6(input_tensor, keep_prob, dims=dims)
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
  dims = ",".join(["%5d" % dim for dim in t.get_shape().as_list()])
  print("%-21s : [%s]" % (t.op.name, dims))


def mbe_conv2d(tensor, n_in, n_out, name, activate=tf.tanh, verbose=True):
  """ A lazy inner function to create a `tf.nn.conv2d` Tensor.

  Args:
    tensor: a Tensor, [index, 1, w, n_in]
    n_in: the number of input channels.
    n_out: the number of output channels.
    name: the name of this layer.
    activate: the activation function, defaults to `tf.tanh`.
    verbose: print the layer if True

  Returns:
    activated: a Tensor of activated `tf.nn.conv2d`.

  """
  with tf.name_scope(name):
    with tf.name_scope("filter"):
      kernel = tf.Variable(
        tf.truncated_normal(
          [1, 1, n_in, n_out], stddev=0.1, seed=SEED, dtype=TF_TYPE),
        name="kernel")
      variable_summaries(kernel)
    conv = tf.nn.conv2d(
      tensor, kernel, [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=CUDA_ON)
    with tf.name_scope("biases"):
      biases = tf.Variable(
        tf.zeros([n_out], dtype=TF_TYPE), name="biases")
      variable_summaries(biases)
    bias = tf.nn.bias_add(conv, biases)
    activated = activate(bias)
    if verbose:
      print_activations(activated)
  return activated


def mbe_nn_m_6(input_tensor, keep_prob, dims=None, verbose=True):
  """
  Return the infered MBE-NN-M deep neural network model with 6 convolutional
  layers.

  Args:
    input_tensor: a 4D Tensor as the input layer, [batch, 1, C(N,k), C(k,2)]
    keep_prob: a float tensor for dropout layers.
    dims: List[int], a list of major dimensions. If None, the default
      [40, 60, 70, 2, 40, 10] will be used.
    verbose: a bool, if True, the layer definitions will be printed.

  Returns:
    y_pred: the estimated result tensor of shape [batch, 1].

  References:
    Alexandrova, A. N. (2016). http://doi.org/10.1021/acs.jctc.6b00994

  """
  if verbose:
    print("-> Inference the MBE-NN-M model ...")
    print("")

  if dims is None:
    dims = [40, 60, 70, 2, 40, 10]

  cnk, ck2 = input_tensor.get_shape().as_list()[:-2]

  # Build the first three MBE layers.
  # The shape of the input data tensor is [n, 1, C(N,k), C(k,2)].
  # To fit Fk, the NN connection is localized in the second dimension, and the
  # layer size of the first dimension is kept fixed. The weights and biases of
  # NN connection are shared among different indices of the first dimension,
  # so that the fitted function form of Fk is kept consistent among different
  # k-body terms. The MBE part is composed of four layers with the following
  # sizes:
  # (C(N,k), C(k,2)) - (C(N,k), 40) - (C(N,k), 70) - (C(N,k), 60) - (C(N,k), 2).
  conv1 = mbe_conv2d(
    input_tensor,
    ck2,
    dims[0],
    "Conv1",
    activate=tf.nn.tanh
  )

  conv2 = mbe_conv2d(
    conv1,
    dims[0],
    dims[1],
    "Conv2",
    activate=tf.nn.tanh
  )
  drop = tf.nn.dropout(conv2, keep_prob=keep_prob, name="drop2")

  conv3 = mbe_conv2d(
    drop,
    dims[1],
    dims[2],
    "Conv3",
    activate=tf.nn.tanh
  )

  conv4 = mbe_conv2d(
    conv3,
    dims[2],
    dims[3],
    "Conv4",
    activate=tf.nn.softplus
  )
  drop = tf.nn.dropout(conv4, keep_prob=keep_prob, name="drop4")

  # Then we build the three mixing layers.
  # The mixing part is used to fit G. Within this part the NN connection is
  # localized in the first dimension, and the size of the second dimension is
  # kept fixed. The parameters of NN connection in this part are shared among
  # different indices of the second dimension. In this work, the mixing part is
  # composed of two layers with the following sizes:
  # (C(N, k), 2) - (40, 2) - (10, 2).
  reshape = tf.reshape(drop, (-1, 1, dims[3], cnk), name="h2dep")

  conv5 = mbe_conv2d(
    reshape,
    cnk,
    dims[4],
    "Conv5",
    activate=tf.nn.softplus
  )
  conv6 = mbe_conv2d(
    conv5,
    dims[4],
    dims[5],
    "Conv6",
    activate=tf.nn.softplus
  )
  drop = tf.nn.dropout(conv6, keep_prob=keep_prob, name="drop6")

  # The last part is used to transform the output of mixing part to a single
  # value, representing the energy. The average-pooling is used, which means
  # that we take the average value of all elements in the matrix of the previous
  # layer as the final output. In this work, the pooling part is composed of one
  # layer of the size:
  # (10, 2) - (1).
  flat = tf.contrib.layers.flatten(drop)
  return tf.reduce_mean(
    flat,
    axis=1,
    name="y_predictions",
    keep_dims=True
  )


def mbe_nn_fc(input_tensor, keep_prob, conv_dims, fc_dims, dropouts,
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
    keep_prob: a float as the keep probability of dropout tensors.
    conv_dims: List[int], the major dims of the conv layers.
    fc_dims: List[int], the size of the dense layers.
    dropouts: List[int], the indices of the layers to add dropouts.
    verbose: a bool.

  Returns:
    y_pred: a Tensor of shape [-1, 1] as the output layer.

  """
  if verbose:
    print("-> Inference the MBE-NN-M-FC model ...")
    print("")

  cnk, ck2 = input_tensor.get_shape().as_list()[-2:]


