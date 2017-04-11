# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody_input
from itertools import repeat
from tensorflow.contrib.layers.python.layers import conv2d, flatten
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from scipy.misc import comb

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of structures to process in a batch.""")
tf.app.flags.DEFINE_boolean('disable_biases', False,
                            """Disable biases for all conv layers.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          """The initial learning rate.""")
tf.app.flags.DEFINE_string('conv_sizes', '60,120,120,60',
                           """Comma-separated integers as the sizes of the 
                           convolution layers.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.


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
  print("%-25s : [%s]" % (tensor.op.name, dims))


def get_number_of_trainable_parameters(verbose=False):
  """
  Return the number of trainable parameters in current graph.

  Args:
    verbose: a bool. If True, the number of parameters for each variable will
    also be printed.

  Returns:
    ntotal: the total number of trainable parameters.

  """
  print("")
  print("Compute the total number of trainable parameters ...")
  print("")
  ntotal = 0
  for var in tf.trainable_variables():
    nvar = np.prod(var.get_shape().as_list(), dtype=int)
    if verbose:
      print("{:25s}   {:d}".format(var.name, nvar))
    ntotal += nvar
  print("Total number of parameters: %d" % ntotal)
  print("")


def inputs(train=True, shuffle=True):
  """
  Construct input for k-body evaluation using the Reader ops.

  Args:
    train: bool, indicating if one should use the train or eval data set.
    shuffle: bool, indicating if the batches shall be shuffled or not.

  Returns:
    features: Behler features for the molecules. 4D tensor of shape
      [batch_size, 1, NATOMS, NDIMS].
    energies: the dedired energies. 2D tensor of shape [batch_size, 1].

  """
  return kbody_input.inputs(
    train=train,
    batch_size=FLAGS.batch_size,
    num_epochs=None,
    shuffle=shuffle
  )


def inputs_settings(train=True):
  """
  Return the dict of global settings for inputs.

  Args:
    train: boolean indicating if one should return the training settings or
      validation settings.

  Returns:
    setting: a dict of settings.

  """
  return kbody_input.inputs_settings(train=train)


def inference_sum_kbody(conv, kbody_term, ck2, sizes=(60, 120, 120, 60),
                        verbose=True):
  """
  Infer the k-body term of `sum-kbody-cnn`.

  Args:
    conv: a `[-1, 1, N, M]` Tensor as the input. N is the number of atoms in
      the molecule and M is the number of features.
    kbody_term: a `str` Tensor as the name of this k-body term.
    ck2: a `int` as the value of C(k,2).
    sizes: a `List[int]` as the number of convolution kernels for each layer.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    kbody_energies: a Tensor of `[-1, 1, N, 1]` as the energy contributions of 
      all kbody terms.

  """

  if verbose:
    print("Infer the %s term of `sum-kbody-cnn` ..." % kbody_term)

  num_layers = len(sizes)
  activation_fn = list(repeat(tf.nn.tanh, num_layers))
  kernel_size = 1
  stride = 1
  padding = 'SAME'
  dtype = tf.float32

  # Explicitly set the shape of the input tensor. There are two flexible axis in
  # this tensor: axis=0 represents the batch size and axis=2 is determined by
  # the number of atoms.
  conv.set_shape([None, 1, None, ck2])

  for i, units in enumerate(sizes):
    if FLAGS.disable_biases:
      biases_initializer = None
    else:
      biases_initializer = init_ops.zeros_initializer(dtype=dtype)
    conv = conv2d(
      conv,
      units,
      kernel_size,
      activation_fn=activation_fn[i],
      stride=stride,
      padding=padding,
      scope="Hidden{:d}".format(i + 1),
      weights_initializer=initializers.xavier_initializer(
        seed=kbody_input.SEED, dtype=dtype),
      biases_initializer=biases_initializer,
    )
    if verbose:
      print_activations(conv)

  kbody_energies = conv2d(
    conv,
    1,
    kernel_size,
    activation_fn=tf.nn.relu,
    biases_initializer=None,
    weights_initializer=initializers.xavier_initializer(
      seed=kbody_input.SEED, dtype=dtype),
    stride=stride,
    padding=padding,
    scope="k-Body"
  )
  if verbose:
    print_activations(kbody_energies)
    print("")

  # Directly return the 4D tensor of kbody energies. The sum/flatten will be
  # done in the main inference function.
  return kbody_energies


def inference(batch_inputs, split_dims, kbody_terms, verbose=True,
              conv_sizes=(60, 120, 120, 60)):
  """
  The general inference function.

  Args:
    batch_inputs: a Tensor of shape `[-1, 1, -1, D]` as the inputs.
    split_dims: a `List[int]` or a 1-D Tensor containing the sizes of each 
      output tensor along split_dim.
    kbody_terms: a `List[str]` as the names of the k-body terms.
    verbose: boolean indicating whether the layers shall be printed or not.
    conv_sizes: a `Tuple[int]` as the sizes of the convolution layers.

  Returns:
    total_energies: a Tensor representing the predicted total energies.
    contribs: a Tensor representing the predicted contributions of the kbody
      terms.

  """

  def zero_padding():
    """
    Return a 4D float32 Tensor of zeros as the fake inputs.
    """
    return tf.zeros([1, 1, 1, 1], name="zeros")

  # Build up the placeholder tensors.
  #   extra_inputs: a placeholder Tensor of shape `[-1, 1, -1, D]` as the
  #     alternative inputs which can be directly feeded.
  #   use_extra: a boolean Tensor indicating whether we should use the
  #     placeholder inputs or the batch inputs.
  #   is_predicting: a boolean Tensor indicating whether the model should be
  #     used for training/evaluation or prediction.
  with tf.name_scope("placeholders"):
    ck2 = int(comb(FLAGS.many_body_k, 2, exact=True))
    use_extra = tf.placeholder_with_default(
      False,
      shape=None,
      name="use_extra_inputs"
    )
    extra_inputs = tf.placeholder_with_default(
      tf.zeros([1, 1, 1, ck2]),
      shape=[None, 1, None, ck2],
      name="extra_inputs"
    )

  # Choose the inputs tensor based on `use_placeholder`.
  with tf.name_scope("InputsFlow"):
    selected = tf.cond(
      use_extra,
      lambda: extra_inputs,
      lambda: batch_inputs,
      name="selected_inputs"
    )

    axis = tf.constant(2, dtype=tf.int32, name="CNK")
    eps = tf.constant(0.001, dtype=tf.float32, name="threshold")
    convs = tf.split(selected, split_dims, axis=axis, name="Partition")

  kbody_energies = []

  for i, conv in enumerate(convs):
    with tf.variable_scope(kbody_terms[i]):
      with tf.variable_scope("Conv"):
        contribs = inference_sum_kbody(
          conv,
          kbody_terms[i],
          ck2,
          sizes=conv_sizes,
          verbose=verbose
        )

      below_zero = tf.less(
        tf.reduce_sum(conv, name="inputs_sum"),
        eps,
        name="below_zero"
      )
      return_zero = tf.logical_and(
        is_predicting,
        below_zero,
        name="check_zero"
      )
      kbody = tf.cond(
        return_zero, zero_padding, lambda: contribs, name="zero_or_conv"
      )
    kbody_energies.append(kbody)

  contribs = tf.concat(kbody_energies, axis=axis, name="Contribs")
  tf.summary.histogram("kbody_contribs", contribs)
  if verbose:
    print_activations(contribs)

  with tf.name_scope("Outputs"):
    total_energies = tf.reduce_sum(contribs, axis=axis, name="Total")
    total_energies.set_shape([None, 1, 1])
    if verbose:
      print_activations(total_energies)

    total_energies = tf.squeeze(flatten(total_energies), name="squeeze")
    if verbose:
      get_number_of_trainable_parameters(verbose=verbose)

  return total_energies, contribs


def get_total_loss(y_true, y_pred):
  """
  Return the total loss tensor.

  Args:
    y_true: the desired energies.
    y_pred: the predicted energies.

  Returns:
    loss: the total loss tensor.

  """
  with tf.name_scope("RMSE"):
    loss = tf.losses.mean_squared_error(y_true, y_pred, scope="sMSE")
    loss = tf.sqrt(loss, name="sRMSE")
    tf.summary.scalar('sRMSE', loss)

  return loss


def get_train_op(total_loss, global_step):
  """
  Train the Behler model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: the total loss Tensor.
    global_step: Integer Variable counting the number of training steps
      processed.

  Returns:
    train_op: op for training.

  """
  # Compute gradients.
  opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
  grads = opt.compute_gradients(total_loss)

  # Add histograms for grandients
  grad_norms = []
  with tf.name_scope("2norms"):
    for grad, var in grads:
      norm = tf.norm(grad, name=var.op.name + "/2norm")
      grad_norms.append(norm)
      tf.summary.histogram(var.op.name + "/gradients", grad)
      tf.summary.scalar(var.op.name + "/grad_norm", norm)
  total_norm = tf.add_n(grad_norms, "total_norms")
  tf.summary.scalar("total_grad_norm", total_norm)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
