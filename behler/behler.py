"""
Inference the Behler's feed-forward neural network.

References:
  * Behler, J. (2011). Phys Chem Chem Phys 13: 17930-17955.
  * Behler, J. (2015). International Journal of Quantum Chemistry,
    115(16): 1032-1050.

The model implemented here is intended for monoatomic clusters.

"""

from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import symmetry
import sys
import re
import time
from tensorflow.contrib.layers.python.layers import conv2d, flatten
from os.path import isfile, join
from sklearn.model_selection import train_test_split


__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of structures to process in a batch.""")
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """The maximum number of training steps.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
MOMENTUM_FACTOR = 0.7

# The number of atoms is 20
NATOMS = 20

# Use the default Behler settings, so the number of features for each atom is 8,
# including 4 radial features and 4 angular features.
NDIMS = 8

# The cutoff radius is set to 6.5 angstroms.
Rc = 6.5

# Use the optimized B20 structures to train this demo
XYZ_FILE = join("..", "datasets", "B20pbe_opted.xyz")

# The total number of training samples.
TOTAL_SIZE = 2400

# Use 20% of the samples as the testing samples.
TEST_SIZE = 0.2

# The random seed
SEED = 218

# Set the npz file to save fingerprints
NPZ_FILE = "B20.npz"


def extract_xyz(filename, verbose=True):
  """
  Extract symbols, coordiantes and forces (for later usage) from the file.

  Args:
    filename: a str, the file to parse.
    verbose: a bool.

  Returns
    species: Array[NUM_SITES], an array of the atomic symbols.
    energies: Array[N,], a 1D array of the atomic energies.
    coordinates: Array[N, 17, 3], a 3D array of the atomic coordinates.

  """
  energies = np.zeros((TOTAL_SIZE,), dtype=np.float32)
  coordinates = np.zeros((TOTAL_SIZE, NATOMS, 3), dtype=np.float32)
  species = []
  parse_species = True
  parse_forces = False
  stage = 0
  i = 0
  j = 0

  energy_patt = re.compile(r"([\w.-]+)")
  string_patt = re.compile(
    r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")

  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")

  with open(filename) as f:
    for line in f:
      if i == TOTAL_SIZE:
        break
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          n = int(l)
          if n != NATOMS:
            raise ValueError("The parsed size %d != NUM_SITES" % n)
          stage += 1
      elif stage == 1:
        m = energy_patt.search(l)
        if m:
          energies[i] = float(m.group(1))
          stage += 1
      elif stage == 2:
        m = string_patt.search(l)
        if m:
          coordinates[i, j, :] = float(m.group(2)), float(m.group(3)), float(
            m.group(4))
          if parse_species:
            species.append(m.group(1))
            if len(species) == NATOMS:
              species = np.asarray(species, dtype=object)
              parse_species = False
          j += 1
          if j == NATOMS:
            j = 0
            stage = 0
            i += 1
            if verbose and i % 1000 == 0:
              sys.stdout.write("\rProgress: %7d  /  %7d" % (i, TOTAL_SIZE))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  return species, energies, coordinates


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
  ntotal = 0
  for var in tf.trainable_variables():
    nvar = np.prod(var.get_shape().as_list(), dtype=int)
    if verbose:
      print("{:25s}  {:d}".format(var.name, nvar))
    ntotal += nvar

  print("")
  print("Total number of parameters: %d" % ntotal)


def inference(features, energies, activation=tf.nn.tanh, hidden_sizes=(10, 10),
              verbose=True):
  """
  Infer the Behler's model for a monoatomic cluster.

  Args:
    features: a `[-1, N, M]` Tensor as the input. N is the number of atoms in
      the monoatomic cluster and M is the number of features.
    energies: a `[-1, 1]` Tensor as the desired energies.
    activation: the activation function. Defaults to `tf.nn.tanh`.
    hidden_sizes: List[int], the number of units of each hidden layer.
    verbose: a bool. If Ture, the shapes of the layers will be printed.

  Returns:
    total_energy: the output Tensor of shape `[-1, 1]` as the estimated total
      energies.
    atomic_energies: a Tensor of `[-1, N]`, the estimated energy for each atom.

  """
  assert len(hidden_sizes) >= 1

  shapes = features.get_shape().as_list()
  natoms, ndims = shapes[1:]
  conv = tf.reshape(features, (-1, 1, natoms, ndims))
  if verbose:
    print_activations(conv)

  kernel_size = 1
  stride = 1
  padding = 'SAME'

  for i, units in enumerate(hidden_sizes):
    conv = conv2d(
      conv,
      units,
      kernel_size,
      activation_fn=activation,
      stride=stride,
      padding=padding,
      scope="Hidden{:d}".format(i + 1)
    )
    if verbose:
      print_activations(conv)

  atomic_energies = conv2d(
    conv,
    1,
    kernel_size,
    activation_fn=None,
    biases_initializer=None,
    stride=stride,
    padding=padding,
    scope="AtomEnergy"
  )
  if verbose:
    print_activations(atomic_energies)

  flat = flatten(atomic_energies)
  total_energy = tf.reduce_sum(flat, axis=1, keep_dims=True)
  if verbose:
    print_activations(total_energy)

  return total_energy, atomic_energies


def may_build_dataset(verbose=True):
  """
  Build the dataset if the npz file cannot be accessed. Then return the energies
  and Behler features.

  Args:
    verbose: a bool, If True the running progress will be printed.

  Returns:
    features: a 3D array as the Behler's training features.
    energies: a 2D array as the desired energies.

  """
  if not isfile(NPZ_FILE):
    _, energies, coordinates = extract_xyz(XYZ_FILE, verbose=verbose)
    ntotal = len(energies)
    features = np.zeros((ntotal, NATOMS, NDIMS), dtype=np.float32)
    energies = energies.astype(np.float32)

    print("Building Behler's fingerprints ...")
    tic = time.time()

    for i in range(ntotal):
      features[i] = symmetry.get_behler_fingerprints(coordinates[i], Rc)
      if verbose and i % 100 == 0:
        sys.stdout.write("\rProgress: %4d / %4d" % (i, ntotal))
    print("Total time: %.3f s" % (time.time() - tic))

    np.savez(NPZ_FILE, features=features, energies=energies)

  else:
    ar = np.load(NPZ_FILE)
    features = ar["features"]
    energies = ar["energies"]

  return features.astype(np.float32), energies.astype(np.float32)


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

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
    tf.summary.scalar(l.op.name + ' (raw)', l)
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

  # Variables that affect learning rate.
  num_batches_per_epoch = TOTAL_SIZE * TEST_SIZE // FLAGS.batch_size
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


def train(*args):
  tf.reset_default_graph()

  # Load the features and energies and than split them into a training dataset
  # and a testing dataset.
  features, energies = may_build_dataset(verbose=True)
  X_train, y_train, X_test, y_test = train_test_split(
    features,
    energies,
    test_size=TEST_SIZE,
    random_state=SEED
  )

  with tf.Graph().as_default():

    # Inference
    X_batch = tf.placeholder(tf.float32, [None, NATOMS, NDIMS], name="X_batch")
    y_batch = tf.placeholder(tf.float32, [None, 1], name="y_batch")
    total_energy, atomic_energies = inference(
      X_batch,
      y_batch,
      hidden_sizes=(100, 100)
    )

    # Setup the loss function
    loss = tf.sqrt(tf.losses.mean_squared_error(y_batch, total_energy),
                   name="RMSE")

    # Setup the optimization
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = get_train_op(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


if __name__ == "__main__":
  tf.app.run(main=train)
