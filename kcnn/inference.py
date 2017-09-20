#!coding=utf-8
"""
This module is used to inference the model of `KCNN`.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
from scipy.misc import comb
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.contrib.layers import conv2d, flatten
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from constants import KcnnGraphKeys, SEED
from utils import lrelu

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The decay for batch normalization
BATCH_NORM_DECAY_FACTOR = 0.999


def _inference_kbody_cnn(inputs, kbody_term, ck2, is_training, verbose=True,
                         reuse=False, normalizer='bias', activation_fn=lrelu,
                         weights_initializer=None, num_kernels=None):
  """
  Infer the k-body term of `KCNN`.

  Args:
    inputs: a `[-1, 1, -1, C(k, 2)]` Tensor as the inputs for this interaction.
    kbody_term: a `str` as the name of this k-body atomic interaction.
    ck2: a `int` as the value of C(k,2).
    is_training: a `bool` type placeholder tensor indicating whether this
      inference is for training or not.
    reuse: a `bool` indicating whether we should reuse the variables or not.
    normalizer: a `str` as the normalizer to use. Supported normalizers include:
      'bias' (default), 'batch_norm', 'layer_norm' and None.
    activation_fn: a `Callable` as the activation function for each conv layer.
    weights_initializer: a `Callable` as the function to intialize weights.
    num_kernels: a `List[int]` as the number of kernels.
    verbose: a `bool`. If Ture, the shapes of the layers will be printed.

  Returns:
    contribs: a Tensor of shape `[-1, 1, N, 1]` as the energy contribs of all
      possible atom combinations.

  """

  if verbose:
    tf.logging.info("Infer the %s term of `KCNN` ..." % kbody_term)

  num_kernels = num_kernels or (40, 50, 60, 40)
  kernel_size = 1
  dtype = tf.float32

  # Explicitly set the shape of the input tensor. There are two flexible axis in
  # this tensor: axis=0 represents the batch size and axis=2 is determined by
  # the number of atoms.
  inputs.set_shape([None, 1, None, ck2])

  # Setup the initializers and normalization function.
  weights_initializer = weights_initializer or initializers.xavier_initializer(
    seed=SEED, dtype=dtype)
  biases_initializer = None
  normalizer_fn = None
  normalizer_params = {}
  collections = [KcnnGraphKeys.FORCES_VARIABLES,
                 KcnnGraphKeys.ENERGY_VARIABLES]

  if normalizer == 'bias':
    biases_initializer = init_ops.zeros_initializer()
  elif normalizer == 'batch_norm':
    normalizer_fn = batch_norm
    normalizer_params = {"decay": BATCH_NORM_DECAY_FACTOR,
                         "scale": True,
                         "center": True,
                         "scope": "bn",
                         "reuse": reuse,
                         "is_training": is_training,
                         "variables_collections": collections}
  elif normalizer == 'layer_norm':
    biases_initializer = init_ops.zeros_initializer()
    normalizer_fn = layer_norm
    normalizer_params = {"scope": "ln",
                         "reuse": reuse,
                         "variables_collections": collections}

  # Build the convolution neural network for this k-body atomic interaction.
  with arg_scope([conv2d],
                 weights_initializer=weights_initializer,
                 normalizer_fn=None,
                 reuse=reuse,
                 variables_collections=collections):
    for i, num_kernels in enumerate(num_kernels):
      with tf.variable_scope("Hidden{:d}".format(i + 1), reuse=reuse):
        inputs = conv2d(inputs,
                        kernel_size=kernel_size,
                        biases_initializer=biases_initializer,
                        num_outputs=num_kernels,
                        activation_fn=None,
                        scope="1x1Conv{:d}".format(i + 1))
        if normalizer_fn is not None:
          inputs = normalizer_fn(inputs, **normalizer_params)
        inputs = activation_fn(inputs)
        if verbose:
          print_activations(inputs)

    outputs = conv2d(inputs,
                     kernel_size=kernel_size,
                     num_outputs=1,
                     biases_initializer=None,
                     activation_fn=None,
                     scope="k-Body")
    if verbose:
      print_activations(outputs)
      tf.logging.info("")
    return outputs


def _inference_1body_nn(occurs, num_atom_types, initial_one_body_weights=None,
                        reuse=False, trainable=True):
  """
  Inference the one-body part.

  Args:
    occurs: a Tensor of shape `[-1, 1, 1, num_atom_types]` as the number of
      occurances of each type of atom.
    num_atom_types: a `int` as the number of atom types.
    initial_one_body_weights: an array of shape `[num_atom_types, ]` as the
      initial weights of the one-body kernel.
    reuse: a `bool` indicating whether we should reuse the variables or not.
    trainable: a `bool` indicating whether the one-body parameters are trainable
      or not.

  Returns:
    one_body: a 4D Tensor of shape `[-1, 1, 1, 1]` as the one-body contribs.

  """
  num_outputs = 1
  kernel_size = 1

  if initial_one_body_weights is not None:
    values = np.asarray(initial_one_body_weights, dtype=np.float32)
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
    reuse=reuse,
    variables_collections=[KcnnGraphKeys.ENERGY_VARIABLES],
    trainable=trainable,
  )


def inference_forces(y_total, inputs, coefficients, indexing):
  """
  Inference the kCON forces.

  Args:
    y_total: a `float32` Tensor of shape `[-1, ]` as the output.
    inputs: a Tensor of shape `[-1, 1, D, C(k, 2)]` as the inputs.
    coefficients: a 3D Tensor as the auxiliary coefficients for computing atomic
      forces.
    indexing: a 3D Tensor as the indexing matrix for force compoenents.

  Returns:
    forces: a `float32` Tensor of shape `[-1, num_force_components]` as the
      neural network forces.

  """

  assert isinstance(coefficients, tf.Tensor)
  assert isinstance(indexing, tf.Tensor)

  with tf.name_scope("Forces"):

    # Compute the derivative of dE / dz. `z` is the input feature matrix and
    # E = NN(z) is the output of the KCNN model.
    dydz = tf.gradients([y_total], [inputs], name="dydz")[0]

    # Squeeze the `dydz`. Now its shape will be `[-1, D, C(k, 2)]`
    dydz = tf.squeeze(dydz, axis=1, name="dydz3d")

    # Tile the derivatives because each entry of `z` contributes to six force
    # components.
    tiled = tf.tile(dydz, (1, 1, 6), "tiled")

    # Do the element-wise multiplication. Now we get dy/dz * dz/dr * dr/df.
    # Here `f` represents arbitrary force compoenent. `g` has the shape of
    # `[-1, D, 6 * C(k, 2)]`.
    g = tf.multiply(tiled, coefficients, name="g")

    # Now we should re-order all entries of `g`. Flatten it so that its shape
    # will be `[-1, D * 6 * C(k, 2)]`.
    g = flatten(g)

    # The basic idea of the re-ordering algorithm is taking advantage of the
    # array broadcasting scheme of TensorFlow (Numpy). Since the batch size (the
    # first axis of `g`) will not be 1, we cannot do broadcasting directly.
    # Instead, we make the `g` a total flatten vector and broadcast it into a
    # matrix with `indexing`.
    with tf.name_scope("reshape"):

      with tf.name_scope("indices"):

        with tf.name_scope("g"):
          shape = tf.shape(g, name="shape")
          batch_size, step = shape[0], shape[1]

        with tf.name_scope("indexing"):
          shape = tf.shape(indexing, name="shape")
          num_f, num_entries = shape[1], shape[2]

        multiples = [1, num_f, num_entries]
        size = tf.multiply(batch_size, step, name="total_size")
        steps = tf.range(0, limit=size, delta=step, name="arange")
        steps = tf.reshape(steps, (batch_size, 1, 1), name="steps")
        steps = tf.tile(steps, multiples, name="tiled")
        indices = tf.add(indexing, steps, name="indices")

      # Do the broadcast
      g = tf.reshape(g, (-1, ), "1D")

      # Pad an zero at the begining of the totally flatten `g` because real
      # indices in `indexing` start from one and the index of zero suggests the
      # contribution should also be zero.
      g = tf.pad(g, [[1, 0]], name="pad")
      g = tf.gather(g, indices, name="gather")

      # Reshape `g` so that all entries of each row (axis=2) correspond to the
      # same force component (axis=1).
      g = tf.reshape(g, (batch_size, num_f, num_entries), "reshape")

    # Sum up all entries of each row to get the final gradient for each force
    # component.
    g = tf.reduce_sum(g, axis=2, keep_dims=False, name="sum")

    # Always remember the physics law: f = -dE / dr. But the output `y_total`
    # already took the minus sign.
    forces = tf.identity(g, "forces")

  return forces


def _split_inputs(inputs, split_dims):
  """
  Split the inputs into different parts. Each part represents a k-body atomic
  interaction.
  """
  with tf.name_scope("Split"):
    axis = tf.constant(2, dtype=tf.int32, name="major")
    return tf.split(inputs, split_dims, axis=axis)


def inference_energy(inputs, occurs, weights, split_dims, num_atom_types,
                     kbody_terms, is_training, max_k=3, reuse=False,
                     verbose=True, num_kernels=None, activation_fn=lrelu,
                     normalizer='bias', weights_initializer=None,
                     one_body_weights=None, trainable_one_body=True,
                     add_summary=True):
  """
  Inference the kCON energy model.

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
    reuse: a `bool` indicating whether we should reuse variables or not.
    verbose: boolean indicating whether the layers shall be printed or not.
    num_kernels: a `Tuple[int]` as the number of kernels of the convolution
      layers. This also determines the number of layers in each atomic network.
    activation_fn: a `Callable` as the activation function.
    normalizer: a `str` as the normalizer to use. Supported normalizers include:
      'bias' (default), 'batch_norm', 'layer_norm' and None.
    weights_initializer: a `Callable` as the function to initialize weights.
    one_body_weights: a `float32` array of shape `[num_atom_types, ]` as the
      initial weights of the one-body kernel.
    trainable_one_body: a `bool` indicating whether the one body parameters are
      trainable or not.
    add_summary: a `bool` indicating whether we should add summaries for
      tensors or not.

  Returns:
    y_total: a `float32` Tensor of shape `[-1, ]` as the total energies.
    y_contribs: a `float32` Tensor of shape `[-1, D]` as the predicted energies
      of the kbody contribs.
    forces: a `float32` Tensor of shape `[-1, num_force_components]` as the
      neural network forces.

  """

  # Split the input feature matrix into several parts. Each part represents a
  # certain atomic interaction. The number of parts is equal to the number of
  # k-body terms.
  num_cols = int(comb(max_k, 2, exact=True))
  splited_inputs = _split_inputs(inputs, split_dims)

  # Inference the convolution network for each k-body interaction
  y_contribs = []
  for i, conv in enumerate(splited_inputs):
    with tf.variable_scope(kbody_terms[i]):
      y_contribs.append(
        _inference_kbody_cnn(
          inputs=conv,
          kbody_term=kbody_terms[i],
          ck2=num_cols,
          activation_fn=activation_fn,
          reuse=reuse,
          normalizer=normalizer,
          weights_initializer=weights_initializer,
          is_training=is_training,
          num_kernels=num_kernels,
          verbose=verbose)
      )

  # Concat the k-body contribs from all k-body terms. The new tensor has the
  # shape of `[-1, 1, D, 1]`.
  contribs = tf.concat(y_contribs, axis=2, name="raw_contribs")

  # Obtain the weighted k-body contribs.
  # In general we hope zero inputs lead to zero contribs. But the convolution
  # kernels have biases so the output may not be zero. To fix this potential
  # problem we multiply the calculated k-body contribs with binary weights.
  contribs = tf.multiply(contribs, weights, name="y_contribs")
  if add_summary:
    tf.summary.histogram("kbody_contribs", contribs)
  if verbose:
    print_activations(contribs)

  # Inference the one-body expression.
  one_body = _inference_1body_nn(occurs,
                                 num_atom_types,
                                 reuse=reuse,
                                 initial_one_body_weights=one_body_weights,
                                 trainable=trainable_one_body)
  if add_summary:
    tf.summary.histogram("1body_contribs", one_body)
  if verbose:
    print_activations(one_body)

  # Sum up the k-body contribs and one-body contribs to get the total energy.
  # This is why we call this network `KCNN`.
  with tf.name_scope("Sum"):
    with tf.name_scope("kbody"):
      y_total_kbody = tf.reduce_sum(contribs, axis=2, name="Total")
      y_total_kbody.set_shape([None, 1, 1])
      if verbose:
        print_activations(y_total_kbody)
      y_total_kbody = tf.squeeze(flatten(y_total_kbody), name="squeeze")
      if add_summary:
        tf.summary.scalar("kbody_mean", tf.reduce_mean(y_total_kbody))
    with tf.name_scope("1body"):
      y_total_1body = tf.squeeze(one_body, name="squeeze")
      if add_summary:
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
  dims = ",".join(["{:16d}".format(dim if dim is not None else -1)
                   for dim in tensor.get_shape().as_list()])
  tf.logging.info("%-36s : [%s]" % (tensor.op.name, dims))


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


def debug():
  """
  Debug the KCNN inference.
  """
  properties = dict(
    num_atom_types=4,
    split_dims=[84, 252, 36, 36, 189, 63, 63, 9, 35, 21, 21, 7],
    atomic_forces=True,
    kbody_terms=["CCC", "CCH", "CCN", "CCX", "CHH", "CHN", "CHX", "CNX", "HHH",
                 "HHN", "HHX", "HNX"],
    one_body_weights=np.ones(4, dtype=np.float32),
  )

  graph = tf.Graph()

  with graph.as_default():

    inputs = tf.placeholder(
      tf.float32, shape=[50, 1, None, 3], name="inputs"
    )
    occurs = tf.placeholder(
      tf.float32, shape=[50, 1, 1, properties['num_atom_types']], name="occurs"
    )
    binary_weights = tf.placeholder(
      tf.float32, shape=[50, 1, None, 1], name="weights"
    )
    split_dims = tf.placeholder(
      tf.int64, shape=[len(properties['split_dims']), 0], name="split_dims"
    )
    coefficients = tf.placeholder(
      tf.float32, shape=[50, None, 18], name="aux_coef"
    )
    indexing = tf.placeholder(
      tf.int32, shape=[50, None, None], name="indexing"
    )

    y_total, y_contribs = inference_energy(
      inputs=inputs,
      occurs=occurs,
      weights=binary_weights,
      split_dims=split_dims,
      num_atom_types=properties['num_atom_types'],
      kbody_terms=properties['kbody_terms'],
      is_training=True,
      verbose=True,
      one_body_weights=properties['one_body_weights'],
    )

    f_calc = inference_forces(
      y_total=y_total,
      inputs=inputs,
      coefficients=coefficients,
      indexing=indexing
    )

    print(y_total.get_shape())
    print(y_contribs.get_shape())
    print(f_calc.get_shape())


if __name__ == "__main__":
  debug()
