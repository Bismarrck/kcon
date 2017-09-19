# coding=utf-8
"""
This script is used to infer the neural network.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import reader
from constants import VARIABLE_MOVING_AVERAGE_DECAY, LOSS_MOVING_AVERAGE_DECAY
from constants import SEED
from inference import inference
from utils import lrelu, selu, selu_initializer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of structures to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          """The learning rate for minimizing energy.""")
tf.app.flags.DEFINE_string('conv_sizes', '40,50,60,40',
                           """Comma-separated integers as the sizes of the 
                           convolution layers.""")
tf.app.flags.DEFINE_string('initial_one_body_weights', None,
                           """Comma-separated floats as the initial one-body 
                           weights. Defaults to `ones_initialier`.""")
tf.app.flags.DEFINE_boolean('fixed_one_body', False,
                            """Make the one-body weights fixed.""")
tf.app.flags.DEFINE_string('activation_fn', "lrelu",
                           """Set the activation function for conv layers.""")
tf.app.flags.DEFINE_float('alpha', 0.01,
                          """Set the parameter `alpha` for `lrelu`.""")
tf.app.flags.DEFINE_string('normalizer', None,
                           """Set the normalizer: 'bias'(default), 'batch_norm', 
                           'layer_norm' or 'None'. """)
tf.app.flags.DEFINE_float('floss_weight', 1.0,
                          """The weight of the f-loss in total loss.""")


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
  elif name.lower() == 'leaky_relu' or name.lower() == 'lrelu':
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
  indices = 7


def get_batch(train=True, shuffle=True, dataset_name=None):
  """
  Construct input for k-body evaluation using the Reader ops.

  Args:
    train: a `bool` indicating if one should use the train or eval data set.
    shuffle: a `bool` indicating if the batches shall be shuffled or not.
    dataset_name: a `str` as the dataset to use.

  Returns:
    features: Behler features for the molecules. 4D tensor of shape
      [batch_size, 1, NATOMS, NDIMS].
    energies: the dedired energies. 2D tensor of shape [batch_size, 1].

  """
  return reader.y_inputs(
    train=train,
    batch_size=FLAGS.batch_size,
    shuffle=shuffle,
    dataset_name=dataset_name
  )


def get_batch_configs(train=True, dataset_name=None):
  """
  Return the configs for inputs.

  Args:
    train: boolean indicating if one should return the training settings or
      validation settings.
    dataset_name: a `str` as the name of the dataset.

  Returns:
    configs: a `dict` as the configs for the dataset.

  """
  return reader.inputs_configs(train=train, dataset_name=dataset_name)


def kcnn(inputs, occurs, weights, split_dims, num_atom_types, kbody_terms,
         is_training, reuse=False, num_kernels=None, verbose=True,
         one_body_weights=None, atomic_forces=False, coefficients=None,
         indexing=None, add_summary=True):
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
  alpha = FLAGS.alpha
  if FLAGS.activation_fn == 'selu':
    weights_initializer = selu_initializer(seed=SEED)
  else:
    weights_initializer = None

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

  y_calc, _, f_calc = inference(
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
    alpha=alpha,
    normalizer=normalizer,
    weights_initializer=weights_initializer,
    one_body_weights=one_body_weights,
    verbose=verbose,
    trainable_one_body=trainable,
    atomic_forces=atomic_forces,
    coefficients=coefficients,
    indexing=indexing,
    add_summary=add_summary,
  )
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


def kcnn_y_from_dataset(dataset_name, for_training=True, **kwargs):
  """
  Inference a kCON energy model based on the given dataset.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` indicating whether this inference is for training or
      evaluation.
    kwargs: additional key-value parameters.

  Returns:
    y_total: a `float32` Tensor of shape `(-1, )` as the predicted total energy.
    y_true: a `float32` Tensor of shape `(-1, )` as the true energy.
    y_weight: a `float32` Tensor of shape `(-1, )` as the weights for computing
      weighted RMSE loss.

  """
  batch = get_batch(
    train=for_training,
    dataset_name=dataset_name,
    shuffle=for_training
  )
  configs = get_batch_configs(train=for_training, dataset_name=dataset_name)
  params = extract_configs(configs, for_training=for_training)
  for key, val in kwargs.items():
    if key in params:
      params[key] = val
    else:
      tf.logging.warning("Unrecognized key={}".format(key))

  y_true = batch[BatchIndex.y_true]
  y_weight = batch[BatchIndex.loss_weight]

  if not params["atomic_forces"]:
    y_total, _ = kcnn(batch[BatchIndex.inputs],
                      batch[BatchIndex.occurs],
                      batch[BatchIndex.weights],
                      **params)

  else:
    raise ValueError("This function only inference energy models!")

  return y_total, y_true, y_weight


def kcnn_yf_from_dataset(dataset_name, for_training=True, **kwargs):
  """
  Inference a kCON energy & forces model for the given dataset.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` indicating whether the model is for training or not.
    **kwargs: additional key-value pairs.

  """

  configs = get_batch_configs(train=for_training, dataset_name=dataset_name)
  params = extract_configs(configs, for_training=for_training)
  for key, val in kwargs.items():
    if key in params:
      params[key] = val
    else:
      tf.logging.warning("Unrecognized key={}".format(key))
  assert params['atomic_forces']

  if for_training:
    feed_batches, handles, dataset_iterators = reader.yf_inputs(
      dataset_name=dataset_name,
      for_training=for_training,
      batch_size=FLAGS.batch_size,
      shuffle=True
    )

    # Inference the energy model
    y_params = dict(params)
    y_params['atomic_forces'] = False
    y_params['reuse'] = False
    with tf.name_scope("Energy"):
      y_calc, _ = kcnn(inputs=feed_batches[0][BatchIndex.inputs],
                       occurs=feed_batches[0][BatchIndex.occurs],
                       weights=feed_batches[0][BatchIndex.weights],
                       **y_params)
      y_true = feed_batches[0][BatchIndex.y_true]
      y_weight = feed_batches[0][BatchIndex.loss_weight]

    # Inference the forces model, `reuse` should be set to True.
    f_params = dict(params)
    f_params['reuse'] = True
    f_params['add_summary'] = False
    with tf.name_scope("Forces"):
      _, f_calc = kcnn(inputs=feed_batches[1][BatchIndex.inputs],
                       occurs=feed_batches[1][BatchIndex.occurs],
                       weights=feed_batches[1][BatchIndex.weights],
                       coefficients=feed_batches[1][BatchIndex.coefficients],
                       indexing=feed_batches[1][BatchIndex.indices],
                       **f_params)
      f_true = feed_batches[1][BatchIndex.f_true]

    # Return the tensors
    calc = {"y": y_calc, "f": f_calc}
    true = {"y": y_true, "f": f_true}
    auxiliary_tensors = {
      "y_weight": y_weight,
      "handles": {
        "y": handles[0],
        "f": handles[1]
      },
      "dataset_iterators": {
        "y": dataset_iterators[0],
        "f": dataset_iterators[1]
      }
    }
    return calc, true, auxiliary_tensors

  else:
    raise NotImplementedError()


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
    mean_squared_error = tf.losses.mean_squared_error(
      y_true, y_nn, scope="MSE", loss_collection=None, weights=weights)
    rmse = tf.sqrt(mean_squared_error, name="RMSE")
    tf.add_to_collection('losses', rmse)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


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
      f_mse = tf.losses.mean_squared_error(
        f_true,
        f_nn,
        scope="fMSE",
        loss_collection=None,
      )
      f_rmse = tf.sqrt(f_mse, name="fRMSE")
      f_loss = tf.multiply(f_rmse, FLAGS.floss_weight, name="f_loss")

    with tf.name_scope("energy"):
      y_mse = tf.losses.mean_squared_error(
        y_true,
        y_nn,
        scope="yMSE",
        loss_collection=None,
      )
      y_loss = tf.sqrt(y_mse, name="yRMSE")

    with tf.name_scope("losses"):
      tf.summary.scalar("y", y_loss)
      tf.summary.scalar("f", f_loss)

    loss = tf.add(f_loss, y_loss, name="together")
    tf.add_to_collection("losses", loss)
    total_loss = tf.add_n(tf.get_collection("losses"), name="total")

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
  with tf.control_dependencies(dependencies):
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
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
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Track the moving averages of all trainable variables.
  with tf.name_scope("average"):
    variable_averages = tf.train.ExponentialMovingAverage(
      VARIABLE_MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
