# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import pipeline
from tensorflow.contrib.opt import NadamOptimizer
from constants import VARIABLE_MOVING_AVERAGE_DECAY, LOSS_MOVING_AVERAGE_DECAY
from constants import SEED
from inference import inference_energy, inference_forces
from utils import lrelu, selu, selu_initializer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of structures to process in a batch.""")

# Setup the structure of the model
tf.app.flags.DEFINE_string('conv_sizes', '40,60,60,40',
                           """Comma-separated integers as the sizes of the 
                           convolution layers.""")
tf.app.flags.DEFINE_string('initial_one_body_weights', None,
                           """Comma-separated floats as the initial one-body 
                           weights. Defaults to `ones_initialier`.""")
tf.app.flags.DEFINE_boolean('fixed_one_body', False,
                            """Make the one-body weights fixed.""")
tf.app.flags.DEFINE_string('activation_fn', "lrelu",
                           """Set the activation function for conv layers.""")
tf.app.flags.DEFINE_string('normalizer', 'bias',
                           """Set the normalizer: 'bias'(default), 'batch_norm', 
                           'layer_norm' or 'None'. """)

# Setup the total loss function
tf.app.flags.DEFINE_float('floss_weight', 1.0,
                          """The weight of the f-loss in total loss.""")
tf.app.flags.DEFINE_boolean('mse', False,
                            """Use MSE loss instead of RMSE loss if True.""")
tf.app.flags.DEFINE_float('l2', None,
                          """Set the lambda of the l2 loss. If None, 
                          l2 regularizer is disabled.""")

# Setup the learning rate. By default the Adam optimizer and constant initial
# learning rate are adopted.
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          """The learning rate for minimizing energy.""")
tf.app.flags.DEFINE_string('learning_rate_decay', None,
                           """The decay function. Default is None. Available
                           options are: exponential, inverse_time and 
                           natrual_exp.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95,
                          """A Python number.  The decay rate.""")
tf.app.flags.DEFINE_float('learning_rate_decay_step', 1000,
                          """How often to apply decay.""")

# Setup the SGD optimizer.
tf.app.flags.DEFINE_string('optimizer', 'adam',
                           """Set the optimizer to use. Avaible optimizers are:
                           adam, nadam, rmsprop, adadelta""")
tf.app.flags.DEFINE_float('beta1', 0.9,
                          """The beta1 of Adam/Nadam""")
tf.app.flags.DEFINE_float('rho', 0.95,
                          """The rho of Adadelta.""")
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9,
                          """The decay factor of RMSProp.""")
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0,
                          """The momentum factor for the RMSProp optimizer.""")


def get_learning_rate(global_step):
  """
  Return the tensor of learning rate.

  Args:
    global_step: an `int64` tensor as the global step.

  Returns:
    lr: a `float32` tensor as the learning rate.

  """
  if FLAGS.learning_rate_decay is None:
    return tf.constant(
      FLAGS.learning_rate, dtype=tf.float32, name="learning_rate")

  else:
    if FLAGS.learning_rate_decay == 'exponential':
      lr = tf.train.exponential_decay
    elif FLAGS.learning_rate_decay == 'inverse_time':
      lr = tf.train.inverse_time_decay
    elif FLAGS.learning_rate_decay == 'natural_exp':
      lr = tf.train.natural_exp_decay
    else:
      raise ValueError(
        'Supported decay functions: exponential, inverse_time, natural_exp')
    learning_rate = lr(FLAGS.learning_rate, global_step=global_step,
                       decay_rate=FLAGS.learning_rate_decay_factor,
                       decay_steps=FLAGS.learning_rate_decay_step,
                       staircase=FLAGS.staircase, name="learning_rate")
    tf.summary.scalar('decayed_learning_rate', learning_rate)
    return learning_rate


def get_optimizer(learning_rate):
  """
  Return the tensor of SGD optimizer.

  Args:
    learning_rate: a `float32` tensor as the learning rate.

  Returns:
    optimizer: an optimizer.

  """

  if FLAGS.optimizer == 'adam':
    return tf.train.AdamOptimizer(
      learning_rate=learning_rate, beta1=FLAGS.beta1)
  elif FLAGS.optimizer == 'nadam':
    return NadamOptimizer(learning_rate=learning_rate, beta1=FLAGS.beta1)
  elif FLAGS.optimzier == 'adadelta':
    return tf.train.AdadeltaOptimizer(
      learning_rate=learning_rate, rho=FLAGS.rho)
  elif FLAGS.optimizer == 'rmsprop':
    return tf.train.RMSPropOptimizer(
      learning_rate=learning_rate,
      decay=FLAGS.rmsprop_decay,
      momentum=FLAGS.rmsprop_momentum)
  else:
    raise ValueError("Supported SGD optimizers: adam, nadam, adadelta, rmsprop")


def get_activation_fn(name='lrelu'):
  """
  Return a callable activation function.

  Args:
    name: a `str` as the name of the activation function.

  Returns:
    fn: a `Callable` as the activation function.

  """
  if name.lower() == 'tanh':
    return tf.nn.tanh
  elif name.lower() == 'relu':
    return tf.nn.relu
  elif name.lower() == 'lrelu':
    return lrelu
  elif name.lower() == 'softplus':
    return tf.nn.softplus
  elif name.lower() == 'sigmoid':
    return tf.nn.sigmoid
  elif name.lower() == 'elu':
    return tf.nn.elu
  elif name.lower() == 'selu':
    return selu
  else:
    raise ValueError("The %s activation is not supported!".format(name))


class BatchIndex:
  """
  The indices for manipulating the tuple from `get_batch`.
  """
  inputs = 0
  y_true = 1
  occurs = 2
  weights = 3
  loss_weight = 4
  f_true = 5
  coefficients = 6
  indexing = 7


def kcnn(inputs, occurs, weights, split_dims=(), num_atom_types=None,
         kbody_terms=(), is_training=True, reuse=False, num_kernels=None,
         verbose=True, one_body_weights=None, atomic_forces=False,
         coefficients=None, indexing=None, add_summary=True):
  """
  Inference the model of `KCNN`.

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
    reuse: a `bool` indicating whether we should reuse variables or not.
    verbose: boolean indicating whether the layers shall be printed or not.
    num_kernels: a `Tuple[int]` as the number of kernels of the convolution
      layers. This also determines the number of layers in each atomic network.
    one_body_weights: a 1D array of shape `[nat, ]` as the initial
      weights of the one-body kernel.
    atomic_forces: a `bool` indicating whether the atomic forces should be
      trained or not.
    coefficients: a 3D Tensor as the auxiliary coefficients for computing atomic
      forces.
    indexing: a 3D Tensor as the indexing matrix for force compoenents.
    add_summary: a `bool` indicating whether we should add summaries for
      tensors or not.

  Returns:
    y_calc: a Tensor of shape `[-1, ]` as the predicted total energies.
    f_calc: a Tensor of shape `[-1, 3N]` as the calculated atomic forces.

  """

  num_kernels = num_kernels or (40, 50, 60, 40)

  activation_fn = get_activation_fn(FLAGS.activation_fn)
  weights_initializer = None

  if FLAGS.activation_fn == 'selu':
    weights_initializer = selu_initializer(seed=SEED)

  trainable = not FLAGS.fixed_one_body
  if FLAGS.normalizer.lower() == 'none':
    normalizer = None
  else:
    normalizer = FLAGS.normalizer

  if FLAGS.initial_one_body_weights is not None:
    one_body_weights = FLAGS.initial_one_body_weights
    one_body_weights = np.array([float(x) for x in one_body_weights.split(",")])
    if num_atom_types > 1 and len(one_body_weights) == 1:
      one_body_weights = np.ones(
        num_atom_types, dtype=np.float32) * one_body_weights[0]

  with tf.name_scope("Energy"):
    y_calc, _ = inference_energy(
      inputs,
      occurs,
      weights,
      split_dims,
      num_atom_types=num_atom_types,
      kbody_terms=kbody_terms,
      is_training=is_training,
      reuse=reuse,
      max_k=FLAGS.k_max,
      num_kernels=num_kernels,
      activation_fn=activation_fn,
      normalizer=normalizer,
      weights_initializer=weights_initializer,
      one_body_weights=one_body_weights,
      verbose=verbose,
      trainable_one_body=trainable,
      add_summary=add_summary,
    )

  if atomic_forces:
    with tf.name_scope("Forces"):
      f_calc = inference_forces(
        y_total=y_calc,
        inputs=inputs,
        coefficients=coefficients,
        indexing=indexing
      )
  else:
    f_calc = None

  return y_calc, f_calc


def extract_configs(configs, for_training=True):
  """
  Extract the config of a dataset.

  Args:
    configs: a `dict` as the configs of a dataset.
    for_training: a `bool` indicating whether the configs should be used for
      training or not.

  Returns:
    params: a `dict` as the parameters for inference.

  """

  # Extract the constant configs from the dict
  split_dims = np.asarray(configs["split_dims"], dtype=np.int64)
  num_atom_types = configs["num_atom_types"]
  kbody_terms = [term.replace(",", "") for term in configs["kbody_terms"]]
  num_kernels = [int(units) for units in FLAGS.conv_sizes.split(",")]
  atomic_forces = configs.get("atomic_forces_enabled", False)

  # The last weight corresponds to the average contribs from k_max-body terms.
  weights = np.array(configs["initial_one_body_weights"], dtype=np.float32)
  if len(weights) == 0:
    weights = np.ones((num_atom_types, ), dtype=np.float32)
  else:
    weights = weights[:num_atom_types]

  # Create the parameter dict and the feed dict
  params = dict(split_dims=split_dims, kbody_terms=kbody_terms,
                is_training=for_training, one_body_weights=weights,
                num_atom_types=num_atom_types, num_kernels=num_kernels,
                atomic_forces=atomic_forces)
  return params


def kcnn_from_dataset(dataset_name, for_training=True, num_epochs=None,
                      **kwargs):
  """
  Inference a kCON model for the given dataset.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` indicating whether this inference is for training or
      evaluation.
    num_epochs: an `int` as the maximum number of epochs to run.
    kwargs: additional key-value parameters.

  Returns:
    y_total: a `float32` Tensor of shape `(-1, )` as the predicted total energy.
    y_true: a `float32` Tensor of shape `(-1, )` as the true energy.
    y_weight: a `float32` Tensor of shape `(-1, )` as the weights for computing
      weighted RMSE loss.

  """
  batch = pipeline.next_batch(
    for_training=for_training,
    dataset_name=dataset_name,
    shuffle=for_training,
    num_epochs=num_epochs,
    batch_size=FLAGS.batch_size,
  )
  configs = pipeline.get_configs(
    for_training=for_training, dataset_name=dataset_name
  )
  params = extract_configs(configs, for_training=for_training)
  for key, val in kwargs.items():
    if key in params:
      params[key] = val
    else:
      tf.logging.warning("Unrecognized key={}".format(key))

  y_true = batch[BatchIndex.y_true]
  y_weight = batch[BatchIndex.loss_weight]
  f_calc = None
  f_true = None

  if not params["atomic_forces"]:
    y_calc, _ = kcnn(
      batch[BatchIndex.inputs],
      batch[BatchIndex.occurs],
      batch[BatchIndex.weights],
      **params
    )
  else:
    y_calc, f_calc = kcnn(
      batch[BatchIndex.inputs],
      batch[BatchIndex.occurs],
      batch[BatchIndex.weights],
      coefficients=batch[BatchIndex.coefficients],
      indexing=batch[BatchIndex.indexing],
      **params,
    )
    f_true = batch[BatchIndex.f_true]

  return y_calc, y_true, y_weight, f_calc, f_true


def get_y_loss(y_true, y_nn, weights=None):
  """
  Return the total loss tensor of energy only.

  Args:
    y_true: a `float32` tensor of shape `[-1, ]` the true energies.
    y_nn: a `float32` tensor of shape `[-1, ]` as the neural network predicted
      energies.
    weights: the weights for the energies.

  Returns:
    loss: a `float32` scalar tensor as the total loss.

  """
  with tf.name_scope("yRMSE"):
    if weights is None:
      weights = tf.constant(1.0, name='weight')
    loss = tf.losses.mean_squared_error(
      y_true, y_nn, scope="MSE", loss_collection=None, weights=weights)
    if not FLAGS.mse:
      loss = tf.sqrt(loss, name="RMSE")
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total')


def get_yf_joint_loss(y_true, y_nn, f_true, f_nn):
  """
  Return the joint total loss tensor.

  Args:
    y_true: a `float32` tensor of shape `[-1, ]` the true energies.
    y_nn: a `float32` tensor of shape `[-1, ]` as the neural network predicted
      energies.
    f_true: a `float32` tensor of shape `[-1, 3N]` as the true forces.
    f_nn: a `float32` tensor of shape `[-1, 3N]` as the neural network predicted
      atomic forces.

  Returns:
    loss: a `float32` scalar tensor as the total loss.
    y_loss: a `float32` scalar tensor as the energy loss.
    f_loss: a `float32` scalar tensor as the forces loss.

  """
  with tf.name_scope("yfRMSE"):

    with tf.name_scope("forces"):
      f_loss = tf.losses.mean_squared_error(
        f_true,
        f_nn,
        scope="fMSE",
        loss_collection=None,
      )
      if not FLAGS.mse:
        f_loss = tf.sqrt(f_loss, name="fRMSE")
      f_loss = tf.multiply(f_loss, FLAGS.floss_weight, name="f_loss")

    with tf.name_scope("energy"):
      y_loss = tf.losses.mean_squared_error(
        y_true,
        y_nn,
        scope="yMSE",
        loss_collection=None,
      )
      if not FLAGS.mse:
        y_loss = tf.sqrt(y_loss, name="yRMSE")

    with tf.name_scope("losses"):
      tf.summary.scalar("y", y_loss)
      tf.summary.scalar("f", f_loss)

    loss = tf.add(f_loss, y_loss, name="together")
    tf.add_to_collection("losses", loss)
    total_loss = tf.add_n(tf.get_collection("losses"), name="total")

    if FLAGS.l2 is not None:
      total_loss = _add_l2_regularizer(total_loss, eta=FLAGS.l2)

    return total_loss, y_loss, f_loss


def _add_loss_summaries(total_loss):
  """Add summaries for losses in KCNN model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  
  """

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(LOSS_MOVING_AVERAGE_DECAY,
                                                    name='avg')
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


def _add_total_norm_summaries(grads_and_vars, collection,
                              only_summary_total=True):
  """
  Add summaries for the 2-norms of the gradients.

  Args:
    grads_and_vars: a list of (gradient, variable) returned by an optimizer.
    collection: a `str` as the collection to add the computed norms.
    only_summary_total: a `bool` indicating whether we should summarize the
      individual norms or not.

  Returns:
    total_norm: a `float32` tensor that computes the sum of all norms of the
      gradients.

  """
  for grad, var in grads_and_vars:
    if grad is not None:
      norm = tf.norm(grad, name=var.op.name + "/norm")
      tf.add_to_collection(collection, norm)
      if not only_summary_total:
        with tf.name_scope("norms/{}/".format(collection)):
          tf.summary.scalar(var.op.name, norm)

  with tf.name_scope("total_norm/"):
    total_norm = tf.add_n(tf.get_collection(collection))
    tf.summary.scalar(collection, total_norm)

  return total_norm


def _add_variable_summaries():
  """
  Add variable summaries.
  """
  with tf.name_scope("variables"):

    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
      vsum = tf.reduce_sum(tf.abs(var, name="absolute"), name="vsum")

      if not var.op.name.startswith('one-body'):
        tf.add_to_collection('vars_k_sum', vsum)
      else:
        tf.add_to_collection('vars_1_sum', vsum)

    tf.summary.scalar(
      'kbody', tf.add_n(tf.get_collection('vars_k_sum'), name='kbody_vars_sum'))
    tf.summary.scalar(
      '1body', tf.add_n(tf.get_collection('vars_1_sum'), name='1body_vars_sum'))


def _add_l2_regularizer(total_loss, eta):
  """
  Add l2 regularizer to the total loss.

  Args:
    total_loss: a `float32` tensor as the total loss.
    eta: a `float32` tensor as the strength of the l2. `eta` is used to replace
      `lambda` in the formula because `lambda` is a Python key word.

  Returns:
    total_loss: a `float32` tensor as the regularized total loss.

  """
  with tf.name_scope("L2"):
    for var in tf.trainable_variables():
      if 'bias' in var.op.name:
        continue
      # L2 loss will not include the one-body weights.
      if var.op.name.startswith('one-body'):
        continue
      l2 = tf.nn.l2_loss(var, name=var.op.name + "/l2")
      tf.add_to_collection('l2_loss', l2)
    l2_loss = tf.add_n(tf.get_collection('l2_loss'), name='l2_raw')
    l2_loss = tf.multiply(l2_loss, eta, name='loss')

  tf.summary.scalar('l2_loss', l2_loss)
  return tf.add(total_loss, l2_loss, name='total_loss_and_l2')


def get_joint_loss_train_op(total_loss, global_step):
  """
  Train the model by minimizing the joint total loss.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: a `float32` Tensor as the joint total loss.
    global_step: Integer Variable counting the number of training steps
      processed.

  Returns:
    train_op: the op for training.

  """

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Add the update ops if batch_norm is True.
  # If we don't include the update ops as dependencies on the train step, the
  # batch_normalization layers won't update their population statistics, which
  # will cause the model to fail at inference time.
  dependencies = [loss_averages_op]
  if FLAGS.normalizer and FLAGS.normalizer == 'batch_norm':
    dependencies.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

  # Compute gradients.
  with tf.name_scope("Optimizer"):
    with tf.control_dependencies(dependencies):
      lr = get_learning_rate(global_step=global_step)
      opt = get_optimizer(learning_rate=lr)
      grads = opt.compute_gradients(total_loss)

  # Add histograms for grandients
  _add_total_norm_summaries(grads, "norms", only_summary_total=False)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(
    grads,
    global_step=global_step,
    name="apply_grads"
  )

  # Add histograms for trainable variables.
  _add_variable_summaries()

  # Track the moving averages of all trainable variables.
  with tf.name_scope("average"):
    variable_averages = tf.train.ExponentialMovingAverage(
      VARIABLE_MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
