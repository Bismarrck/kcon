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
from kbody_input import SEED

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of structures to process in a batch.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0        # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.98  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1        # Initial learning rate.


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


def _inference_alex_kbody(conv, kbody_term, sizes=(40, 70, 60, 2, 40, 20),
                          verbose=True):
  """
  Infer the k-body term of `alex-kbody-cnn`.

  Args:
    conv: a `[-1, 1, N, M]` Tensor as the input. N is the number of atoms in
      the molecule and M is the number of features.
    kbody_term: a `str` Tensor as the name of this k-body term.
    sizes: a `List[int]` as the major dimensions of the layers.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    energies: a Tensor of shape `[-1, 1]` as the estimated k-body energies.

  """

  with tf.variable_scope(kbody_term):

    if verbose:
      print("Infer the %s term of `alex-kbody-cnn` ..." % kbody_term)

    axis_dim = conv.get_shape().as_list()[2]
    num_layers = len(sizes)
    activation_fn = list(repeat(tf.nn.tanh, 4)) + \
                    list(repeat(tf.nn.softplus, num_layers - 4))
    kernel_size = 1
    stride = 1
    padding = 'SAME'
    dtype = kbody_input.get_float_type(convert=True)

    for i in range(num_layers):
      conv = conv2d(
        conv,
        sizes[i],
        kernel_size,
        stride,
        padding,
        activation_fn=activation_fn[i],
        weights_initializer=init_ops.truncated_normal_initializer(
          stddev=0.1, dtype=dtype, seed=SEED),
        biases_initializer=init_ops.zeros_initializer(dtype=dtype),
        scope="Conv{:d}".format(i + 1)
      )
      if verbose:
        print_activations(conv)
      if i == 3:
        conv = tf.reshape(conv, [-1, 1, sizes[i], axis_dim], name="switch")
        if verbose:
          print_activations(conv)

    flat = flatten(conv, scope="kbody")
    if verbose:
      print_activations(flat)

    energies = tf.reduce_mean(flat, axis=1, keep_dims=True, name="total")
    if verbose:
      print_activations(energies)

    return energies


def _inference_sum_kbody(conv, kbody_term, sizes=(40, 60, 80, 70, 10),
                         verbose=True):
  """
  Infer the k-body term of `sum-kbody-cnn`.

  Args:
    conv: a `[-1, 1, N, M]` Tensor as the input. N is the number of atoms in
      the molecule and M is the number of features.
    kbody_term: a `str` Tensor as the name of this k-body term.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    total_energy: the output Tensor of shape `[-1, 1]` as the estimated total
      energies.
    atomic_energies: a Tensor of `[-1, N]`, the estimated energy for each atom.

  """

  with tf.variable_scope(kbody_term):

    if verbose:
      print("Infer the %s term of `sum-kbody-cnn` ..." % kbody_term)

    num_layers = len(sizes)
    activation_fn = list(repeat(tf.nn.tanh, num_layers))
    kernel_size = 1
    stride = 1
    padding = 'SAME'
    dtype = kbody_input.get_float_type(convert=True)

    for i, units in enumerate(sizes):
      conv = conv2d(
        conv,
        units,
        kernel_size,
        activation_fn=activation_fn[i],
        stride=stride,
        padding=padding,
        scope="Hidden{:d}".format(i + 1),
        weights_initializer=initializers.xavier_initializer(
          seed=SEED, dtype=dtype),
        biases_initializer=init_ops.zeros_initializer(dtype=dtype),
      )
      if verbose:
        print_activations(conv)

    k_body_energies = conv2d(
      conv,
      1,
      kernel_size,
      activation_fn=None,
      biases_initializer=None,
      weights_initializer=init_ops.truncated_normal_initializer(
        seed=SEED, dtype=dtype, stddev=0.1),
      stride=stride,
      padding=padding,
      scope="k-Body"
    )
    if verbose:
      print_activations(k_body_energies)

    flat = flatten(k_body_energies)
    if verbose:
      print_activations(flat)
      print("")

  return flat


def inference(input_tensor, offsets, kbody_terms, model='sum-kbody-cnn',
              verbose=True):
  """
  The general inference function.

  Args:
    input_tensor: a Tensor of shape `[-1, 1, H, D]`.
    offsets: a `List[int]` as the offsets to split the input tensor.
    kbody_terms: a `List[str]` as the names of the k-body terms.
    model: a `str` as the name of the model to infer.
    verbose: boolean indicating whether the layers shall be printed or not.

  Returns:
    total_energies: a Tensor representing the estimated total energies.

  """

  offsets = tf.constant(offsets, name="Offsets")
  convs = tf.split(input_tensor, offsets, axis=2, name="Partition")
  kbody_energies = []

  if model.lower() == 'sum-kbody-cnn':
    for i, conv in enumerate(convs):
      kbody = _inference_sum_kbody(conv, kbody_terms[i], verbose=verbose)
      kbody_energies.append(kbody)

  elif model.lower() == 'alex-kbody-cnn':
    for i, conv in enumerate(convs):
      kbody = _inference_alex_kbody(conv, kbody_terms[i], verbose=verbose)
      kbody_energies.append(kbody)

  else:
    raise ValueError("The model `{}` is not supported!".format(model))

  concat = tf.concat(kbody_energies, axis=1)
  if verbose:
    print_activations(concat)

  total_energies = tf.reduce_sum(concat, axis=1, keep_dims=False, name="Total")
  if verbose:
    print_activations(total_energies)

  if verbose:
    get_number_of_trainable_parameters(verbose=verbose)

  return total_energies


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from `get_total_loss()`.
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def get_total_loss(energies, pred_energies, atomic=False):
  """
  Return the total loss tensor.

  Args:
    energies: the desired energies.
    pred_energies: the predicted energies.

  Returns:
    loss: the total loss tensor.

  """
  if atomic:
    scale = tf.constant(FLAGS.num_atoms, tf.float32)
  else:
    scale = tf.constant(1.0, tf.float32)

  with tf.name_scope("outputs"):
    with tf.name_scope("batch"):
      tf.summary.scalar('mean', tf.reduce_mean(energies))
    with tf.name_scope("pred"):
      tf.summary.scalar('mean', tf.reduce_mean(pred_energies))
      tf.summary.histogram('histgram', pred_energies)

  with tf.name_scope("RMSE"):
    loss = tf.losses.mean_squared_error(
      energies / scale,
      pred_energies / scale,
      scope="sMSE"
    )
    loss = tf.sqrt(loss, name="sRMSE")
    tf.summary.scalar('sRMSE', loss)

  return loss


def get_train_op(total_loss, global_step, scale=1.0, clip=False):
  """
  Train the Behler model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: the total loss Tensor.
    global_step: Integer Variable counting the number of training steps
      processed.
    num_kernels: a Tensor of `int` as the total number of 1x1 kernels.
    clip: boolena indicating explicitly clip gradients or not.

  Returns:
    train_op: op for training.

  """

  # Variables that affect learning rate.
  train_size = FLAGS.num_examples * 0.8
  num_batches_per_epoch = train_size // FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Compute gradients.
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(total_loss)

  # Compute the 2-norm of the gradients
  # norms = {}
  # for grad, var in grads:
  #   norms[var] = tf.norm(grad, name=var.op.name + '/gnorm')
  # norm = tf.add_n(list(norms.values()), name="gnorm")
  # scale = tf.minimum(tf.maximum(scale, 1.0 / norm), 1.0)
  # tf.summary.scalar('grad_scale', scale)

  # Explicitly scale the gradients.
  # grads = [(grad * scale, var) for grad, var in grads]

  # Explicitly clip the gradients to avoid explodes!
  if clip:
    grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms and norms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
      # tf.summary.scalar(var.op.name + '/gnorm', norms[var])

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
