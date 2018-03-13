# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import NadamOptimizer

import pipeline
from constants import SEED
from constants import VARIABLE_MOVING_AVERAGE_DECAY, LOSS_MOVING_AVERAGE_DECAY
from summary_utils import add_total_norm_summaries, add_variable_summaries
from inference import inference_energy, inference_forces
from utils import lrelu, selu, reduce_l2_norm
from utils import selu_initializer, msra_initializer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of structures to process in a batch.""")

# Setup the structure of the model
tf.app.flags.DEFINE_string('conv_sizes', '128,64,32',
                           """Comma-separated integers as the sizes of the 
                           convolution layers.""")
tf.app.flags.DEFINE_string('initial_one_body_weights', None,
                           """Comma-separated floats as the initial one-body 
                           weights. Defaults to `ones_initialier`.""")
tf.app.flags.DEFINE_boolean('fixed_one_body', False,
                            """Make the one-body weights fixed.""")
tf.app.flags.DEFINE_integer("trainable_k_max", 3,
                            """Set the trainable k_max.""")
tf.app.flags.DEFINE_string('activation_fn', "lrelu",
                           """Set the activation function for conv layers.""")
tf.app.flags.DEFINE_float('alpha', 0.01,
                          """The alpha value of the leaky relu.""")
tf.app.flags.DEFINE_string('normalizer', 'bias',
                           """Set the normalizer: 'bias'(default), 'batch_norm', 
                           'layer_norm' or 'None'. """)
tf.app.flags.DEFINE_string('initializer', 'msra',
                           """Set the weights initialization method: msra or 
                           xvaier""")

# Setup the total loss function
tf.app.flags.DEFINE_float('f_loss_weight', 1.0,
                          """The weight of the f-loss in total loss.""")
tf.app.flags.DEFINE_float('y_loss_weight', 1.0,
                          """The weight of the y-loss in total loss.""")
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
tf.app.flags.DEFINE_boolean('staircase', False,
                            """Boolean.  If `True` decay the learning rate at 
                            discrete intervals""")
tf.app.flags.DEFINE_float('min_learning_rate', None,
                          """Setup the minimum learning rate.""")

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
    learning_rate = tf.constant(
      FLAGS.learning_rate, dtype=tf.float32, name="raw_learning_rate")
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
                       staircase=FLAGS.staircase, name="raw_lr")

  min_learning_rate = tf.constant(
    FLAGS.min_learning_rate or 0.0, name="minimum_lr")
  learning_rate = tf.maximum(learning_rate, min_learning_rate, "adjusted_lr")
  tf.summary.scalar('lr', learning_rate)
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
    y_calc: a tensor of shape `(-1, )` as the predicted total energy.
    f_calc: a tensor of shape `(-1, 3 * num_atoms)` as the predicted forces.
    num_atoms: a tensor of shape `(-1, )` as the total number of atoms for each
      configuration.

  """

  num_kernels = num_kernels or (40, 50, 60, 40)

  activation_fn = get_activation_fn(FLAGS.activation_fn)
  if FLAGS.activation_fn == 'lrelu':
    activation_fn = partial(activation_fn, alpha=FLAGS.alpha)

  weights_initializer = None

  if FLAGS.activation_fn == 'selu':
    weights_initializer = selu_initializer(seed=SEED)
  elif FLAGS.initializer == 'msra':
    weights_initializer = msra_initializer(seed=SEED)

  trainable = not FLAGS.fixed_one_body
  trainable_k_max = FLAGS.trainable_k_max

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

  with tf.variable_scope("kCON"):
    y_calc, _ = inference_energy(
      inputs,
      occurs,
      weights,
      split_dims,
      num_atom_types=num_atom_types,
      kbody_terms=kbody_terms,
      is_training=is_training,
      reuse=reuse,
      num_kernels=num_kernels,
      activation_fn=activation_fn,
      normalizer=normalizer,
      weights_initializer=weights_initializer,
      one_body_weights=one_body_weights,
      verbose=verbose,
      trainable_one_body=trainable,
      trainable_k_max=trainable_k_max,
      summary=add_summary,
    )

    if atomic_forces:
      f_calc = inference_forces(
        y_total=y_calc,
        inputs=inputs,
        coefficients=coefficients,
        indexing=indexing,
        summary=add_summary
      )
    else:
      f_calc = None

    n_atom = tf.squeeze(
      tf.reduce_sum(occurs, axis=-1, name="sum_of_atoms"), name="num_atoms")
    n_atom.set_shape([None])

  return y_calc, f_calc, n_atom


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
                atomic_forces=atomic_forces, verbose=True)
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
    y_calc: a `float32` Tensor of shape `(-1, )` as the predicted total energy.
    y_true: a `float32` Tensor of shape `(-1, )` as the true energy.
    y_weight: a `float32` Tensor of shape `(-1, )` as the weights for computing
      weighted RMSE loss.
    f_calc: a `float32` Tensor of shape `(-1, -1)` as the predicted atomic
      forces.
    f_true: a `float32` Tensor of shape `(-1, -1)` as the true atomic forces.
    n_atom: a tensor of shape `(-1, )` as the total number of atoms for each
      configuration.

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
    y_calc, _, n_atom = kcnn(
      batch[BatchIndex.inputs],
      batch[BatchIndex.occurs],
      batch[BatchIndex.weights],
      **params
    )
  else:
    y_calc, f_calc, n_atom = kcnn(
      batch[BatchIndex.inputs],
      batch[BatchIndex.occurs],
      batch[BatchIndex.weights],
      coefficients=batch[BatchIndex.coefficients],
      indexing=batch[BatchIndex.indexing],
      **params,
    )
    f_true = batch[BatchIndex.f_true]

  return y_calc, y_true, y_weight, f_calc, f_true, n_atom


def add_l2_regularizer(total_loss, eta):
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


def get_y_loss(y_true, y_calc, weights=None):
  """
  Return the total loss tensor of energy only.

  Args:
    y_true: a `float32` tensor of shape `[-1, ]` as the true energies.
    y_calc: a `float32` tensor of shape `[-1, ]` as the computed energies.
    weights: the weights for the energies.

  Returns:
    loss: a `float32` scalar tensor as the total loss.

  """
  return _get_rmse_loss(y_true, y_calc, weights=weights, scope="yRMSE")


def get_f_loss(f_true, f_calc):
  """
  Return the total loss tensor of forces only.

  Args:
    f_true: a `float32` tensor of shape `[-1, -1]` as the true forces.
    f_calc: a `float32` tensor of shape `[-1, -1]` as the computed forces.

  Returns:
    f_loss: a `float32` scalar tensor as the total loss.

  """
  return _get_rmse_loss(f_true, f_calc, scope="fRMSE", summary_norms=True)


def _get_rmse_loss(true, calc, weights=None, scope=None, summary_norms=False):
  """
  Return the total loss tensor.

  Args:
    true: a `float32` tensor as the true values.
    calc: a `float32` tensor as the computed values. This tensor must has the
      same shape with `true`.
    weights: a tensor as the weights for the mean squared errors.
    scope: a `str` as the name scope.
    summary_norms: a `bool` indicating whether we should summary the norms of
      the values or not.

  Returns:
    total_loss: a `float32` scalar tensor as the total loss.

  """

  with tf.name_scope(scope):
    if weights is None:
      weights = tf.constant(1.0, name='weight')
    loss = tf.losses.mean_squared_error(
      true, calc, scope="MSE", loss_collection=None, weights=weights)
    if not FLAGS.mse:
      loss = tf.sqrt(loss, name="RMSE")
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total')

    if FLAGS.l2 is not None:
      total_loss = add_l2_regularizer(total_loss, eta=FLAGS.l2)

    if summary_norms:
      l2_true = reduce_l2_norm(true, name="l2_true")
      l2_calc = reduce_l2_norm(calc, name="l2_calc")
      with tf.name_scope("magnitudes"):
        tf.summary.scalar('true', l2_true)
        tf.summary.scalar('calc', l2_calc)

    return total_loss


def get_yf_joint_loss(y_true, y_calc, f_true, f_calc):
  """
  Return the joint total loss tensor.

  Args:
    y_true: a `float32` tensor of shape `[-1, ]` the true energies.
    y_calc: a `float32` tensor of shape `[-1, ]` as the energies.
    f_true: a `float32` tensor of shape `[-1, -1]` as the true forces.
    f_calc: a `float32` tensor of shape `[-1, -1]` as the computed forces.

  Returns:
    loss: a `float32` scalar tensor as the total loss.
    y_loss: a `float32` scalar tensor as the energy loss.
    f_loss: a `float32` scalar tensor as the forces loss.

  """
  with tf.name_scope("yfRMSE"):

    with tf.name_scope("forces"):
      f_loss = tf.losses.mean_squared_error(
        f_true,
        f_calc,
        scope="fMSE",
        loss_collection=None,
      )
      if not FLAGS.mse:
        f_loss = tf.sqrt(f_loss, name="fRMSE")
      f_loss = tf.multiply(f_loss, FLAGS.f_loss_weight, name="f_loss")

      with tf.name_scope("magnitudes"):
        tf.summary.scalar('true', reduce_l2_norm(f_true, name="l2_true"))
        tf.summary.scalar('calc', reduce_l2_norm(f_calc, name="l2_calc"))

    with tf.name_scope("energy"):
      y_loss = tf.losses.mean_squared_error(
        y_true,
        y_calc,
        scope="yMSE",
        loss_collection=None,
      )
      if not FLAGS.mse:
        y_loss = tf.sqrt(y_loss, name="yRMSE")
      y_loss = tf.multiply(y_loss, FLAGS.y_loss_weight, name="y_loss")

    with tf.name_scope("losses"):
      tf.summary.scalar("y", y_loss)
      tf.summary.scalar("f", f_loss)

    loss = tf.add(f_loss, y_loss, name="together")
    tf.add_to_collection("losses", loss)
    total_loss = tf.add_n(tf.get_collection("losses"), name="total")

    if FLAGS.l2 is not None:
      total_loss = add_l2_regularizer(total_loss, eta=FLAGS.l2)

    return total_loss, y_loss, f_loss


def get_amp_yf_joint_loss(y_true, y_calc, f_true, f_calc, n_atom):
  """
  The Amp joint loss function.

  Args:
    y_true: a `float32` tensor of shape `(-1, )` the true energies.
    y_calc: a `float32` tensor of shape `(-1, )` as the energies.
    f_true: a `float32` tensor of shape `(-1, -1)` as the true forces.
    f_calc: a `float32` tensor of shape `(-1, -1)` as the computed forces.
    n_atom: a `int32` tensor of shape `(-1, )` as the number of atoms for each
      configuration.

  Returns:
    loss: a `float32` scalar tensor as the total loss.
    y_loss: a `float32` scalar tensor as the energy loss.
    f_loss: a `float32` scalar tensor as the forces loss.

  """

  with tf.name_scope("yf"):

    with tf.name_scope("energy"):
      y_true = tf.divide(y_true, n_atom, name="true/per_atom")
      y_calc = tf.divide(y_calc, n_atom, name="calc/per_atom")
      y_diff = tf.squared_difference(y_true, y_calc, name="squared")
      y_loss = tf.reduce_sum(y_diff, name='loss')

    with tf.name_scope("forces"):
      alpha = tf.constant(1.0, dtype=tf.float32, name="alpha")
      f_diff = tf.squared_difference(f_true, f_calc, name="squared")
      f_loss = tf.reduce_sum(tf.reduce_mean(f_diff, axis=1))
      f_loss = tf.multiply(f_loss, alpha, name="loss")

      with tf.name_scope("magnitudes"):
        tf.summary.scalar('true', reduce_l2_norm(f_true, name="l2_true"))
        tf.summary.scalar('calc', reduce_l2_norm(f_calc, name="l2_calc"))

    with tf.name_scope("losses"):
      tf.summary.scalar('y', y_loss)
      tf.summary.scalar('f', f_loss)

    loss = tf.multiply(0.5, tf.add(y_loss, f_loss, name="add"), name="loss")
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name="total")

    if FLAGS.l2 is not None:
      total_loss = add_l2_regularizer(total_loss, eta=FLAGS.l2)

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
  add_total_norm_summaries(grads, "joint", only_summary_total=False)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(
    grads,
    global_step=global_step,
    name="apply_grads"
  )

  # Add histograms for trainable variables.
  add_variable_summaries()

  # Track the moving averages of all trainable variables.
  with tf.name_scope("average"):
    variable_averages = tf.train.ExponentialMovingAverage(
      VARIABLE_MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def sum_of_gradients(grads_of_losses):
  """
  Calculate the total gradient from `grad_and_vars` of different losses.

  Args:
    grads_of_losses: a list of lists of (gradient, variable) tuples.

  Returns:
    List of pairs of (gradient, variable) as the total gradient.

  """
  # Merge gradients
  sum_grads = []
  for grad_and_vars in zip(*grads_of_losses):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is None:
        continue

      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    v = grad_and_vars[0][1]

    # If the grads are all None, we just return a None grad.
    if len(grads) == 0:
      grad_and_var = (None, v)

    else:
      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_sum(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      grad_and_var = (grad, v)
    sum_grads.append(grad_and_var)

  return sum_grads


def get_yf_train_op(total_loss, y_loss, f_loss, global_step):
  """
  An alternative implementation of building the `train_op`.

  Args:
    total_loss: a `float32` scalar tensor as the total loss. `total_loss` should
      be the sum of `y_loss` and `f_loss`. In this function `total_loss` is used
      for summary only.
    y_loss: a `float32` scalar tensor as the energy loss.
    f_loss: a `float32` scalar tensor as the forces loss.
    global_step: Integer Variable counting the number of training steps
      processed.

  Returns:
    train_op: the op for training.

  """
  total_loss_op = _add_loss_summaries(total_loss)

  # Add the update ops if batch_norm is True.
  # If we don't include the update ops as dependencies on the train step, the
  # batch_normalization layers won't update their population statistics, which
  # will cause the model to fail at inference time.
  dependencies = [total_loss_op]
  if FLAGS.normalizer and FLAGS.normalizer == 'batch_norm':
    dependencies.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

  with tf.name_scope("Optimizer"):
    with tf.control_dependencies(dependencies):
      lr = get_learning_rate(global_step=global_step)
      opt = get_optimizer(learning_rate=lr)
      assert isinstance(opt, tf.train.Optimizer)

  with tf.name_scope("Gradients"):
    with tf.name_scope("y"):
      dydw = opt.compute_gradients(y_loss)
      add_total_norm_summaries(dydw, "y")

    with tf.name_scope("f"):
      dfdw = opt.compute_gradients(f_loss)
      add_total_norm_summaries(dfdw, "f")

  # Merge the gradients
  sum_grads = sum_of_gradients((dydw, dfdw))
  add_total_norm_summaries(sum_grads, "yf", only_summary_total=False)

  # Apply the gradients to variables
  apply_gradient_op = opt.apply_gradients(
    sum_grads, global_step=global_step, name="apply_merged"
  )

  # Add histograms for trainable variables.
  add_variable_summaries()

  # Track the moving averages of all trainable variables.
  with tf.name_scope("average"):
    variable_averages = tf.train.ExponentialMovingAverage(
      VARIABLE_MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def test_sum_of_gradients():
  """
  Test the function `sum_of_gradients`.
  """
  with tf.Graph().as_default():
    a = tf.get_variable('a', shape=(), dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    b = tf.get_variable('b', shape=(), dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    x = tf.placeholder(tf.float32, shape=(None, ), name='x')
    y = tf.placeholder(tf.float32, shape=(None, ), name='y')

    f_calc = a * x + b * tf.pow(y, 2)
    f_true = tf.placeholder(tf.float32, shape=(None, ), name='f_true')
    f_loss = tf.sqrt(tf.losses.mean_squared_error(f_true, f_calc))
    g_calc = b * y
    g_true = tf.placeholder(tf.float32, shape=(None, ), name='g_true')
    g_loss = tf.sqrt(tf.losses.mean_squared_error(g_true, g_calc))
    h_loss = f_loss + g_loss

    data = [[2.0, 3.0, 31.0, 9.0],
            [1.0, 1.0, 5.0, 3.0]]
    data = np.array(data, dtype=np.float32)

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    with tf.name_scope("h"):
      raw_grads = opt.compute_gradients(h_loss)

    with tf.name_scope("separated"):
      with tf.name_scope("f"):
        dfdx = opt.compute_gradients(f_loss)
      with tf.name_scope("g"):
        dgdx = opt.compute_gradients(g_loss)
      sum_grads = sum_of_gradients((dfdx, dgdx))

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      feed_dict = {x: data[:, 0],
                   y: data[:, 1],
                   f_true: data[:, 2],
                   g_true: data[:, 3]}
      grad1 = sess.run(raw_grads, feed_dict=feed_dict)
      grad2 = sess.run(sum_grads, feed_dict=feed_dict)
      assert abs(grad1[0][0] - grad2[0][0]) < 0.001
      assert abs(grad1[1][0] - grad2[1][0]) < 0.001
