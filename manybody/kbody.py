# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody_input
from tensorflow.contrib.layers.python.layers import conv2d, flatten
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from kbody_input import SEED

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of structures to process in a batch.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01      # Initial learning rate.
MOMENTUM_FACTOR = 0.7


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
  Construct input for Behler evaluation using the Reader ops.

  Args:
    train: bool, indicating if one should use the train or eval data set.
    shuffle: bool, indicating if the batches shall be shuffled or not.

  Returns:
    features: Behler features for the molecules. 4D tensor of shape
      [batch_size, 1, NATOMS, NDIMS].
    energies: the dedired energies. 2D tensor of shape [batch_size, 1].

  """
  features, energies, offsets = kbody_input.inputs(
    train=train,
    batch_size=FLAGS.batch_size,
    num_epochs=None,
    shuffle=shuffle)
  return features, energies, offsets


def _inference(conv, verbose=True):
  """
  Infer the Behler's model for a monoatomic cluster.

  Args:
    conv: a `[-1, 1, N, M]` Tensor as the input. N is the number of atoms in
      the monoatomic cluster and M is the number of features.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    total_energy: the output Tensor of shape `[-1, 1]` as the estimated total
      energies.
    atomic_energies: a Tensor of `[-1, N]`, the estimated energy for each atom.

  """

  hidden_sizes = (40, 100, 200, 100, 20)
  hidden_fn = (
    tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.softplus, tf.nn.softplus
  )
  kernel_size = 1
  stride = 1
  padding = 'SAME'
  dtype = kbody_input.get_float_type(convert=True)

  for i, units in enumerate(hidden_sizes):
    conv = conv2d(
      conv,
      units,
      kernel_size,
      activation_fn=hidden_fn[i],
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
    weights_initializer=initializers.xavier_initializer(
      seed=SEED, dtype=dtype),
    stride=stride,
    padding=padding,
    scope="k-Body"
  )
  if verbose:
    print_activations(k_body_energies)

  flat = flatten(k_body_energies)
  if verbose:
    print_activations(flat)

  total_energies = tf.reduce_sum(flat, axis=1, keep_dims=False)
  if verbose:
    print_activations(total_energies)

  if verbose:
    get_number_of_trainable_parameters(verbose=verbose)

  return total_energies, k_body_energies


def inference(input_tensor, offsets, verbose=True):
  """
  The general inference function.

  Args:
    input_tensor:
    offsets:
    verbose:

  Returns:
    total_energies: a Tensor representing the estimated total energies.

  """

  input_convs = tf.split(input_tensor, offsets, axis=2, name="Partition")

  total_energies = []
  kbody_energies = []
  scopes = ["BBBB", "BBBTa"]

  for i, conv in enumerate(input_convs):
    with tf.variable_scope(scopes[i]):
      net_total_energies, net_kbody_energies = _inference(conv, verbose=verbose)
      total_energies.append(net_total_energies)
      kbody_energies.append(net_kbody_energies)

  return tf.multiply(tf.add_n(total_energies), -1.0)


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


def get_total_loss(energies, pred_energies):
  """
  Return the total loss tensor.

  Args:
    energies: the desired energies.
    pred_energies: the predicted energies.

  Returns:
    loss: the total loss tensor.

  """
  with tf.name_scope("RMSE"):
    loss = tf.losses.mean_squared_error(energies, pred_energies, scope="RMS")
    loss = tf.sqrt(loss, name="SQRT")
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

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr, MOMENTUM_FACTOR)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
