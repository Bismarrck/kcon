# coding=utf-8
"""
This script is used to building training and validation datasets.
"""

from __future__ import print_function, absolute_import

import json
import numpy as np
import tensorflow as tf
import transformer
from functools import partial
from os import makedirs
from os.path import join, isfile, isdir
from database import Database

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


tf.app.flags.DEFINE_string("binary_dir", "./binary",
                           """The directory for storing binary datasets.""")
tf.app.flags.DEFINE_string('dataset', 'C9H7N.PBE',
                           """Define the dataset to use. This is also the name
                           of the xyz file to load.""")
tf.app.flags.DEFINE_string('format', 'xyz',
                           """The format of the xyz file.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """The total number of examples to use.""")
tf.app.flags.DEFINE_integer('k_max', 3,
                            """The maximum k under the many-body-expansion 
                            scheme.""")
tf.app.flags.DEFINE_boolean('include_all_k', True,
                            """Include all k-body terms from k = 1 to k_max.""")
tf.app.flags.DEFINE_float('test_size', 0.2,
                          """The proportion of the dataset to include in the 
                          test split""")
tf.app.flags.DEFINE_integer("norm_order", 1,
                            """The exponential order for normalizing 
                            distances.""")
tf.app.flags.DEFINE_boolean('run_input_test', False,
                            """Run the input unit test if True.""")
tf.app.flags.DEFINE_float('unit', None,
                          """Override the default unit if this is not None.""")
tf.app.flags.DEFINE_boolean('periodic', False,
                            """The isomers are periodic structures.""")
tf.app.flags.DEFINE_float('weighted_loss', None,
                          """The kT (eV) for computing the weighted loss. """)

FLAGS = tf.app.flags.FLAGS


def exp_rmse_loss_fn(x, x0=0.0, beta=1.0):
  """
  An exponential function for computing the weighted loss.

  I.e. \\(y = \e^{-\beta \cdot (x - x_0)}\\).
  """
  return np.float32(np.exp(-(x - x0) * beta))


def get_filenames(train=True, dataset=None):
  """
  Return the binary data file and the config file.

  Args:
    train: boolean indicating whether the training data file or the testing file
      should be returned.
    dataset: a `str` as the name of the dataset.

  Returns:
    (binfile, jsonfile): the binary file to read data and the json file to read
      configs..

  """

  binary_dir = FLAGS.binary_dir
  if not isdir(binary_dir):
    makedirs(binary_dir)

  fname = dataset or FLAGS.dataset
  records = {"train": (join(binary_dir, "%s-train.tfrecords" % fname),
                       join(binary_dir, "%s-train.json" % fname)),
             "test": (join(binary_dir, "%s-test.tfrecords" % fname),
                      join(binary_dir, "%s-test.json" % fname))}

  if train:
    return records['train']
  else:
    return records['test']


def may_build_dataset(dataset=None, verbose=True):
  """
  Build the dataset if needed.

  Args:
    verbose: boolean indicating whether the building progress shall be printed
      or not.
    dataset: a `str` as the name of the dataset.

  """
  train_file, _ = get_filenames(train=True, dataset=dataset)
  test_file, _ = get_filenames(train=False, dataset=dataset)

  # Check if the xyz file is accessible.
  xyzfile = join("..", "datasets", "{}.xyz".format(FLAGS.dataset))
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  # Extract the xyz file and split it into two sets: a training set and a
  # testing set.
  database = Database.from_xyz(xyzfile,
                               num_examples=FLAGS.num_examples,
                               verbose=verbose,
                               xyz_format=FLAGS.format)
  database.split(test_size=FLAGS.test_size)

  # The maximum supported `k` is 5.
  k_max = min(5, FLAGS.k_max)

  # Determine the maximum occurances of each atom type.
  max_occurs = database.max_occurs

  # Setup the exponential-scaled RMSE.
  if FLAGS.weighted_loss is None:
    exp_rmse_fn = None
  else:
    min_ener, _ = database.energy_range
    beta = 1.0 / FLAGS.weighted_loss
    exp_rmse_fn = partial(exp_rmse_loss_fn, x0=min_ener, beta=beta)

  # Use a `FixedLenMultiTransformer` to generate features because it will be
  # much easier if the all input samples are fixed-length.
  clf = transformer.FixedLenMultiTransformer(
    max_occurs=max_occurs,
    k_max=k_max,
    periodic=FLAGS.periodic,
    norm_order=FLAGS.norm_order,
    include_all_k=FLAGS.include_all_k,
  )
  clf.transform_and_save(
    database,
    train_file=train_file,
    test_file=test_file,
    verbose=True,
    loss_fn=exp_rmse_fn
  )


def read_and_decode(filename_queue, cnk, ck2, num_atom_types):
  """
  Read and decode the mixed binary dataset file.

  Args:
    filename_queue: an input queue.
    cnk: a `int` as the value of C(N,k). 
    ck2: a `int` as the value of C(k,2). 
    num_atom_types: a `int` as the number of atom types.

  Returns:
    features: a `float32` tensor of shape `(1, cnk, ck2)` as an input matrix.
    energy: a `float64` tensor as the true energy.
    occurs: a `float32` tensor of shape `(1, 1, num_atom_types)`.
    weights: a `float32` tensor of shape `[1, cnk, 1]` as the weights of the
      k-body contribs.
    y_weight: a `float32` tensor as the weight of the loss of this sample.

  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  example = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
      'features': tf.FixedLenFeature([], tf.string),
      'energy': tf.FixedLenFeature([], tf.string),
      'occurs': tf.FixedLenFeature([], tf.string),
      'weights': tf.FixedLenFeature([], tf.string),
      'loss_weight': tf.FixedLenFeature([], tf.float32)
    })

  features = tf.decode_raw(example['features'], tf.float32)
  features.set_shape([cnk * ck2])
  features = tf.reshape(features, [1, cnk, ck2])

  energy = tf.decode_raw(example['energy'], tf.float64)
  energy.set_shape([1])
  energy = tf.squeeze(energy)

  occurs = tf.decode_raw(example['occurs'], tf.float32)
  occurs.set_shape([num_atom_types])
  occurs = tf.reshape(occurs, [1, 1, num_atom_types])

  weights = tf.decode_raw(example['weights'], tf.float32)
  weights.set_shape([cnk, ])
  weights = tf.reshape(weights, [1, cnk, 1])

  y_weight = tf.cast(example['loss_weight'], tf.float32)

  return features, energy, occurs, weights, y_weight


def inputs(train, batch_size=50, shuffle=True, dataset=None):
  """
  Reads mixed input data.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    shuffle: boolean indicating whether the batches shall be shuffled or not.
    dataset: a `str` as the name of the dataset.

  Returns:
    batch: a `Batch`.
      * features: a `float` tensor with shape [batch_size, 1, C(N,k), C(k,2)] in
        the range [0.0, 1.0].
      * energies: a `float` tensor with shape [batch_size, ].
      * weights: a `float` tensor with shape [batch_size, C(N,k)].
      * occurs: a `float` tensor with shape [batch_size, num_atom_types].
      * loss_weight: a `float`.

    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

  """

  filename, _ = get_filenames(train=train, dataset=dataset)
  filenames = [filename]

  configs = inputs_configs(train=train, dataset=dataset)
  shape = configs["shape"]
  cnk = shape[0]
  ck2 = shape[1]
  num_atom_types = configs["num_atom_types"]

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      filenames,
    )

    # Even when reading in multiple threads, share the filename
    # queue.
    features, y_true, occurs, weights, y_weight = read_and_decode(
      filename_queue, cnk, ck2, num_atom_types
    )

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if not shuffle:
      batches = tf.train.batch(
        [features, y_true, occurs, weights, y_weight],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size
      )
    else:
      batches = tf.train.shuffle_batch(
        [features, y_true, occurs, weights, y_weight],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000
      )
    return batches


def inputs_configs(train=True, dataset=None):
  """
  Return the configs for inputs.

  Args:
    train: boolean indicating if one should return settings for training or
      validation.
    dataset: a `str` as the name of the dataset.

  Returns:
    configs: a `dict` of configs.

  """
  _, cfgfile = get_filenames(train=train, dataset=dataset)
  with open(cfgfile) as f:
    return dict(json.load(f))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  if FLAGS.periodic and (FLAGS.format != 'grendel'):
    tf.logging.error(
      "The xyz format must be `grendel` if `periodic` is True!")
  else:
    may_build_dataset(verbose=True)


if __name__ == "__main__":
  tf.app.run(main=main)