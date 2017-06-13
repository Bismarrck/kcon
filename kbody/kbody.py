# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody_input
from itertools import repeat
from tensorflow.contrib.layers.python.layers import conv2d, flatten, batch_norm
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
tf.app.flags.DEFINE_string('initial_one_body_weights', None,
                           """Comma-separated floats as the initial one-body 
                           weights. Defaults to `ones_initialier`.""")
tf.app.flags.DEFINE_boolean('use_linear_output', True,
                            """Set this to True to use linear outputs.""")
tf.app.flags.DEFINE_boolean('fixed_one_body', False,
                            """Make the one-body weights fixed.""")
tf.app.flags.DEFINE_boolean('xavier', True,
                            """Use the xavier algorithm to initialize weights 
                            if this is True. Otherwise the truncated normal 
                            will be used.""")
tf.app.flags.DEFINE_boolean('batch_norm', False,
                            """Use batch normalization if True.""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


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


def inference_sum_kbody(conv, kbody_term, ck2, is_training,
                        sizes=(60, 120, 120, 60), verbose=True):
  """
  Infer the k-body term of `sum-kbody-cnn`.

  Args:
    conv: a `[-1, 1, N, M]` Tensor as the input. N is the number of atoms in
      the molecule and M is the number of features.
    kbody_term: a `str` Tensor as the name of this k-body term.
    ck2: a `int` as the value of C(k,2).
    is_training: a `bool` type placeholder tensor indicating whether this 
      inference is for training or not.
    sizes: a `List[int]` as the number of convolution kernels for each layer.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    kbody_energies: a Tensor of `[-1, 1, N, 1]` as the energy contributions of 
      all kbody terms.

  """

  if verbose:
    tf.logging.info("Infer the %s term of `sum-kbody-cnn` ..." % kbody_term)

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

  # Setup the initializers and normalization function.
  if FLAGS.xavier:
    weights_initializer = initializers.xavier_initializer(
      seed=kbody_input.SEED, dtype=dtype)
  else:
    weights_initializer = init_ops.truncated_normal_initializer(
      seed=kbody_input.SEED, dtype=dtype)

  if FLAGS.disable_biases:
    biases_initializer = None
  else:
    biases_initializer = init_ops.zeros_initializer(dtype=dtype)

  if FLAGS.batch_norm:
    normalizer_fn = batch_norm
    normalizer_params = dict(is_training=is_training)
  else:
    normalizer_fn = None
    normalizer_params = {}

  # Build the network for this block
  for i, n_units in enumerate(sizes):
    conv = conv2d(
      conv,
      n_units,
      kernel_size,
      activation_fn=activation_fn[i],
      stride=stride,
      padding=padding,
      scope="Hidden{:d}".format(i + 1),
      weights_initializer=weights_initializer,
      biases_initializer=biases_initializer,
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params,
    )
    if verbose:
      print_activations(conv)

  kbody_energies = conv2d(
    conv,
    1,
    kernel_size,
    activation_fn=None if FLAGS.use_linear_output else tf.nn.relu,
    biases_initializer=None,
    weights_initializer=weights_initializer,
    stride=stride,
    padding=padding,
    scope="k-Body"
  )
  if verbose:
    print_activations(kbody_energies)
    tf.logging.info("")

  # Directly return the 4D tensor of kbody energies. The sum/flatten will be
  # done in the main inference function.
  return kbody_energies


def inference_one_body(batch_occurs, nat, initial_one_body_weights=None):
  """
  Inference the one-body part.
  
  Args:
    batch_occurs: a 4D Tensor of shape `[-1, 1, 1, Nat]` as the occurances of 
      the atom types.
    nat: a `int` as the number of atom types.
    initial_one_body_weights: a 1D array of shape `[nat, ]` as the initial 
      weights of the one-body kernel.

  Returns:
    one_body: a 4D Tensor of shape `[-1, 1, 1, 1]` as the one-body contribs.

  """
  num_outputs = 1
  kernel_size = 1

  if FLAGS.initial_one_body_weights is not None:
    weights = FLAGS.initial_one_body_weights
    values = [float(x) for x in weights.split(",")]
    if nat > 1 and len(values) == 1:
      values = np.ones(nat, dtype=np.float32) * values[0]
  elif initial_one_body_weights is not None:
    values = initial_one_body_weights
  else:
    values = np.ones(nat, dtype=np.float32)
  if len(values) != nat:
    raise Exception("The number of initial weights should be %d!" % nat)

  weights_initializer = init_ops.constant_initializer(values)

  return conv2d(
    batch_occurs,
    num_outputs=num_outputs,
    kernel_size=kernel_size,
    activation_fn=None,
    weights_initializer=weights_initializer,
    biases_initializer=None,
    scope='one-body',
    trainable=(not FLAGS.fixed_one_body),
  )


def inference(batch_inputs, batch_occurs, batch_weights, split_dims, nat,
              kbody_terms, is_training, verbose=True,
              conv_sizes=(60, 120, 120, 60), initial_one_body_weights=None):
  """
  The general inference function.

  Args:
    batch_inputs: a Tensor of shape `[-1, 1, -1, D]` as the inputs.
    batch_occurs: a Tensor of shape `[-1, Nat]` as the occurances of the atom 
      types. `Nat` is the number of atom types.
    batch_weights: a Tensor of shape `[-1, -1]` as the weights of the k-body 
      contribs.
    split_dims: a `List[int]` or a 1-D Tensor containing the sizes of each 
      output tensor along split_dim.
    nat: a `int` as the number of atom types.
    kbody_terms: a `List[str]` as the names of the k-body terms.
    is_training: a `bool` type placeholder indicating whether this inference is
      for training or not.
    verbose: boolean indicating whether the layers shall be printed or not.
    conv_sizes: a `Tuple[int]` as the sizes of the convolution layers.
    initial_one_body_weights: a 1D array of shape `[nat, ]` as the initial 
      weights of the one-body kernel.

  Returns:
    total_energies: a Tensor representing the predicted total energies.
    contribs: a Tensor representing the predicted contributions of the kbody
      terms.

  """

  # Build up the placeholder tensors.
  #   extra_inputs: a placeholder Tensor of shape `[-1, 1, -1, D]` as the
  #     alternative inputs which can be directly feeded.
  #   use_extra: a boolean Tensor indicating whether we should use the
  #     placeholder inputs or the batch inputs.

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
    extra_weights = tf.placeholder_with_default(
      tf.zeros([1, 1, 1, 1]),
      shape=[None, 1, None, 1],
      name="extra_weights"
    )
    extra_occurs = tf.placeholder_with_default(
      tf.zeros([1, 1, 1, nat]),
      shape=[None, 1, 1, nat],
      name="extra_occurs"
    )

  # Choose the inputs tensor based on `use_placeholder`.
  with tf.name_scope("InputsFlow"):
    selected_inputs, selected_occurs, selected_weights = tf.cond(
      use_extra,
      lambda: (extra_inputs, extra_occurs, extra_weights),
      lambda: (batch_inputs, batch_occurs, batch_weights),
      name="selected"
    )

    axis = tf.constant(2, dtype=tf.int32, name="CNK")
    convs = tf.split(selected_inputs, split_dims, axis=axis, name="Partition")

  kbody_energies = []

  for i, conv in enumerate(convs):
    with tf.variable_scope(kbody_terms[i]):
      kbody = inference_sum_kbody(
        conv,
        kbody_terms[i],
        ck2,
        is_training=is_training,
        sizes=conv_sizes,
        verbose=verbose,
      )
    kbody_energies.append(kbody)

  contribs = tf.concat(kbody_energies, axis=axis, name="Contribs")
  contribs = tf.multiply(contribs, selected_weights, name="Weighted")
  tf.summary.histogram("kbody_contribs", contribs)
  if verbose:
    print_activations(contribs)

  one_body = inference_one_body(selected_occurs, nat, initial_one_body_weights)
  tf.summary.histogram("1body_contribs", one_body)
  if verbose:
    print_activations(one_body)

  with tf.name_scope("Outputs"):
    with tf.name_scope("kbody"):
      y_total_kbody = tf.reduce_sum(contribs, axis=axis, name="Total")
      y_total_kbody.set_shape([None, 1, 1])
      if verbose:
        print_activations(y_total_kbody)
      y_total_kbody = tf.squeeze(flatten(y_total_kbody), name="squeeze")
      tf.summary.scalar("kbody_mean", tf.reduce_mean(y_total_kbody))

    with tf.name_scope("1body"):
      y_total_1body = tf.squeeze(one_body, name="squeeze")
      tf.summary.scalar('1body_mean', tf.reduce_mean(y_total_1body))

    y_total = tf.add(y_total_1body, y_total_kbody, "MBE")

  if verbose:
    get_number_of_trainable_parameters(verbose=verbose)
  return y_total, contribs


def loss(y_true, y_pred, weights=None):
  """
  Return the total loss tensor.

  Args:
    y_true: the desired energies.
    y_pred: the predicted energies.
    weights: the loss weights for the energies.

  Returns:
    loss: the total loss tensor.

  """
  with tf.name_scope("RMSE"):
    if weights is None:
      weights = tf.constant(1.0, name='weight')
    mean_squared_error = tf.losses.mean_squared_error(
      y_true, y_pred, scope="MSE", loss_collection=None, weights=weights)
    rmse = tf.sqrt(mean_squared_error, name="RMSE")
    tf.add_to_collection('losses', rmse)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in sum-kbody-cnn model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  
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
    tf.summary.scalar(l.op.name + '_raw', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


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

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Add the update ops if batch_norm is True.
  # If we don't include the update ops as dependencies on the train step, the
  # batch_normalization layers won't update their population statistics, which
  # will cause the model to fail at inference time
  dependencies = [loss_averages_op]
  if FLAGS.batch_norm:
    dependencies.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

  # Compute gradients.
  with tf.control_dependencies(dependencies):
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = opt.compute_gradients(total_loss)

  # Add histograms for grandients
  with tf.name_scope("gnorms") as scope:
    for grad, var in grads:
      norm = tf.norm(grad, name=var.op.name + "/gnorm")
      tf.add_to_collection('gnorms', norm)
      tf.summary.scalar(var.op.name + "/grad_norm", norm)
    total_norm = tf.add_n(tf.get_collection('gnorms', scope), "total_norms")
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
