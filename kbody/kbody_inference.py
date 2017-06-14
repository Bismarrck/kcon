#!coding=utf-8
"""
This module is used to inference the model of `sum-kbody-cnn`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np

from scipy.misc import comb
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, conv2d, flatten
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from utils import lrelu


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The constant seed for initializing weights.
WEIGHT_INIT_SEED = 218


def inference_kbody_cnn(inputs, kbody_term, ck2, is_training,
                        activation_fn=lrelu, num_kernels=None, verbose=True):
  """
  Infer the k-body term of `sum-kbody-cnn`.

  Args:
    inputs: a `[-1, 1, -1, C(k, 2)]` Tensor as the inputs for this interaction.
    kbody_term: a `str` as the name of this k-body atomic interaction.
    ck2: a `int` as the value of C(k,2).
    is_training: a `bool` type placeholder tensor indicating whether this
      inference is for training or not.
    activation_fn: a `Callable` as the activation function for each conv layer.
    num_kernels: a `List[int]` as the number of kernels.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    contribs: a Tensor of shape `[-1, 1, N, 1]` as the energy contribs of all
      possible atom combinations.

  """

  if verbose:
    tf.logging.info("Infer the %s term of `sum-kbody-cnn` ..." % kbody_term)

  num_kernels = num_kernels or (40, 50, 60, 40)
  kernel_size = 1
  dtype = tf.float32

  # Explicitly set the shape of the input tensor. There are two flexible axis in
  # this tensor: axis=0 represents the batch size and axis=2 is determined by
  # the number of atoms.
  inputs.set_shape([None, 1, None, ck2])

  # Setup the initializers and normalization function.
  weights_initializer = initializers.xavier_initializer(
    seed=WEIGHT_INIT_SEED, dtype=dtype)
  if FLAGS.batch_norm:
    normalizer_fn = batch_norm
  else:
    normalizer_fn = None
  batch_norm_params = {
    "is_training": is_training,
    "decay": BATCH_NORM_DECAY_FACTOR,
    "param_initializers": {
      "gamma_initializer": init_ops.constant_initializer(0.5),
      "moving_variance_initializer": init_ops.constant_initializer(0.5)
    }
  }

  # Build the convolution neural network for this k-body atomic interaction.
  with arg_scope([conv2d],
                 kernel_size=kernel_size,
                 weights_initializer=weights_initializer,
                 normalizer_params=batch_norm_params):
    for i, num_kernels in enumerate(num_kernels):
      inputs = conv2d(inputs,
                      num_outputs=num_kernels,
                      activation_fn=activation_fn,
                      scope="Hidden{:d}".format(i + 1),
                      normalizer_fn=normalizer_fn)
      if verbose:
        print_activations(inputs)

    inputs = conv2d(inputs, num_outputs=1, activation_fn=None,
                    biases_initializer=None, scope="k-Body")
    if verbose:
      print_activations(inputs)
      tf.logging.info("")
    return inputs


def inference_1body_nn(occurs, num_atom_types, initial_one_body_weights=None):
  """
  Inference the one-body part.

  Args:
    occurs: a Tensor of shape `[-1, 1, 1, num_atom_types]` as the number of
      occurances of each type of atom.
    num_atom_types: a `int` as the number of atom types.
    initial_one_body_weights: an array of shape `[num_atom_types, ]` as the
      initial weights of the one-body kernel.

  Returns:
    one_body: a 4D Tensor of shape `[-1, 1, 1, 1]` as the one-body contribs.

  """
  num_outputs = 1
  kernel_size = 1

  if initial_one_body_weights is not None:
    values = initial_one_body_weights
  else:
    values = np.ones(num_atom_types, dtype=np.float32)
  if len(values) != num_atom_types:
    raise Exception("The number of weights should be %d!" % num_atom_types)

  weights_initializer = init_ops.constant_initializer(values)
  return conv2d(
    occurs,
    num_outputs=num_outputs,
    kernel_size=kernel_size,
    activation_fn=None,
    weights_initializer=weights_initializer,
    biases_initializer=None,
    scope='one-body',
  )


def _split_inputs(inputs, split_dims):
  """
  Split the inputs into different parts. Each part represents a k-body atomic
  interaction.
  """
  with tf.name_scope("Split"):
    axis = tf.constant(2, dtype=tf.int32, name="major")
    return tf.split(inputs, split_dims, axis=axis)


def inference(inputs, occurs, weights, split_dims, num_atom_types, kbody_terms,
              is_training, max_k=3, verbose=True, num_kernels=None,
              activation_fn=lrelu, one_body_weights=None):
  """
  The general inference function.

  Args:
    inputs: a Tensor of shape `[-1, 1, -1, D]` as the inputs.
    occurs: a Tensor of shape `[-1, num_atom_types]` as the number of occurances
      of each type of atom.
    weights: a Tensor of shape `[-1, -1, D, -1]` as the weights of the k-body
      contribs.
    split_dims: a `List[int]` or a 1D Tensor of `int` as the dims to split the
      input feature matrix.
    num_atom_types: a `int` as the number of atom types.
    kbody_terms: a `List[str]` as the names of the k-body terms.
    is_training: a `bool` type placeholder indicating whether this inference is
      for training or not.
    max_k: a `int` as the
    verbose: boolean indicating whether the layers shall be printed or not.
    num_kernels: a `Tuple[int]` as the number of kernels of the convolution
      layers. This also determines the number of layers in each atomic network.
    activation_fn: a `Callable` as the activation function.
    one_body_weights: a 1D array of shape `[nat, ]` as the initial
      weights of the one-body kernel.

  Returns:
    y_total: a `float32` Tensor of shape `[-1, ]` as the total energies.
    y_contribs: a `float32` Tensor of shape `[-1, D]` as the predicted energies
      of the kbody contribs.

  """

  # Split the input feature matrix into several parts. Each part represents a
  # certain atomic interaction. The number of parts is equal to the number of
  # k-body terms.
  num_cols = int(comb(max_k, 2, exact=True))
  splited_inputs, = _split_inputs(inputs, split_dims)

  # Inference the convolution network for each k-body interaction
  y_contribs = []
  for i, conv in enumerate(splited_inputs):
    with tf.variable_scope(kbody_terms[i]):
      y_contribs.append(inference_kbody_cnn(conv, kbody_terms[i], num_cols,
                                            activation_fn=activation_fn,
                                            is_training=is_training,
                                            num_kernels=num_kernels,
                                            verbose=verbose))

  # Concat the k-body contribs from all k-body terms. The new tensor has the
  # shape of `[-1, 1, D, 1]`.
  contribs = tf.concat(y_contribs, axis=2, name="Contribs")

  # Obtain the weighted k-body contribs.
  # In general we hope zero inputs lead to zero contribs. But the convolution
  # kernels have biases so the output may not be zero. To fix this potential
  # problem we multiply the calculated k-body contribs with binary weights.
  contribs = tf.multiply(contribs, weights, name="Weighted")
  tf.summary.histogram("kbody_contribs", contribs)
  if verbose:
    print_activations(contribs)

  # Inference the one-body expression.
  one_body = inference_1body_nn(occurs,
                                num_atom_types,
                                initial_one_body_weights=one_body_weights)
  tf.summary.histogram("1body_contribs", one_body)
  if verbose:
    print_activations(one_body)

  # Sum up the k-body contribs and one-body contribs to get the total energy.
  # This is why we call this network `sum-kbody-cnn`.
  with tf.name_scope("Sum"):
    with tf.name_scope("kbody"):
      y_total_kbody = tf.reduce_sum(contribs, axis=2, name="Total")
      y_total_kbody.set_shape([None, 1, 1])
      if verbose:
        print_activations(y_total_kbody)
      y_total_kbody = tf.squeeze(flatten(y_total_kbody), name="squeeze")
      tf.summary.scalar("kbody_mean", tf.reduce_mean(y_total_kbody))
    with tf.name_scope("1body"):
      y_total_1body = tf.squeeze(one_body, name="squeeze")
      tf.summary.scalar('1body_mean', tf.reduce_mean(y_total_1body))
    y_total = tf.add(y_total_1body, y_total_kbody, "1_and_k")

  if verbose:
    get_number_of_trainable_parameters(verbose=verbose)
  return y_total, contribs


def print_activations(tensor):
  """
  Print the name and shape of the input Tensor.

  Args:
    tensor: a Tensor.

  """
  dims = ",".join(["{:7d}".format(dim if dim is not None else -1)
                   for dim in tensor.get_shape().as_list()])
  tf.logging.info("%-25s : [%s]" % (tensor.op.name, dims))


def get_number_of_trainable_parameters(verbose=False):
  """
  Return the number of trainable parameters in current graph.

  Args:
    verbose: a bool. If True, the number of parameters for each variable will
    also be printed.

  Returns:
    ntotal: the total number of trainable parameters.

  """
  tf.logging.info("")
  tf.logging.info("Compute the total number of trainable parameters ...")
  tf.logging.info("")
  ntotal = 0
  for var in tf.trainable_variables():
    nvar = np.prod(var.get_shape().as_list(), dtype=int)
    if verbose:
      tf.logging.info("{:<38s}   {:>8d}".format(var.name, nvar))
    ntotal += nvar
  tf.logging.info("Total number of parameters: %d" % ntotal)
  tf.logging.info("")


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