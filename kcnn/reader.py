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
from collections import namedtuple

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
tf.app.flags.DEFINE_boolean('forces', False,
                            """Set this to True to enable atomic forces.""")
tf.app.flags.DEFINE_boolean('run_test', False,
                            """Run the test.""")

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
  if FLAGS.forces and FLAGS.format != 'ase':
    raise ValueError("Currently only ASE-generated xyz files are supported if "
                     "forces training is enabled.")

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
    atomic_forces=FLAGS.forces
  )
  clf.transform_and_save(
    database,
    train_file=train_file,
    test_file=test_file,
    verbose=True,
    loss_fn=exp_rmse_fn
  )


"""
These namedtuples are just used to manage the decoded example.
"""
EnergyExample = namedtuple("EnergyExample", (
  "features",
  "energy",
  "occurs",
  "weights",
  "y_weight",
))

ForcesExample = namedtuple("ForcesExample", (
  "features",
  "energy",
  "occurs",
  "weights",
  "y_weight",
  "forces",
  "coef",
  "indexing"
))


def decode_protobuf(example_proto, cnk=None, ck2=None, num_atom_types=None,
                    atomic_forces=False, num_f_components=None,
                    num_entries=None):
  """
  Decode the protobuf into a tuple of tensors.

  Args:
    example_proto: A scalar string Tensor, a single serialized Example.
      See `_parse_single_example_raw` documentation for more details.
    cnk: an `int` as the value of C(N,k).
    ck2: an `int` as the value of C(k,2).
    num_atom_types: an `int` as the number of atom types.
    atomic_forces: a `bool` indicating whether atomic forces should be included
      or not.
    num_f_components: an `int` as the maximum number of force components. This
      must be set if `atomic_forces` is True.
    num_entries: an `int` as the number of entries per each force component.
      This must be set if `atomic_forces` is True.

  Returns:
    example: a decoded `TFExample` from the TFRecord file.

  """
  if not atomic_forces:
    example = tf.parse_single_example(
      example_proto,
      # Defaults are not specified since both keys are required.
      features={
        'features': tf.FixedLenFeature([], tf.string),
        'energy': tf.FixedLenFeature([], tf.string),
        'occurs': tf.FixedLenFeature([], tf.string),
        'weights': tf.FixedLenFeature([], tf.string),
        'loss_weight': tf.FixedLenFeature([], tf.float32)
      })
  else:
    example = tf.parse_single_example(
      example_proto,
      features={
        'features': tf.FixedLenFeature([], tf.string),
        'energy': tf.FixedLenFeature([], tf.string),
        'occurs': tf.FixedLenFeature([], tf.string),
        'weights': tf.FixedLenFeature([], tf.string),
        'loss_weight': tf.FixedLenFeature([], tf.float32),
        'coef': tf.FixedLenFeature([], tf.string),
        'indexing': tf.FixedLenFeature([], tf.string),
        'forces': tf.FixedLenFeature([], tf.string)
      })
    assert num_f_components > 0 and num_entries > 0

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

  if atomic_forces:
    coef = tf.decode_raw(example['coef'], tf.float32)
    coef.set_shape([cnk * ck2 * 6])
    coef = tf.reshape(coef, [cnk, ck2 * 6])

    indexing = tf.decode_raw(example['indexing'], tf.int32)
    indexing.set_shape([num_f_components * num_entries])
    indexing = tf.reshape(indexing, [num_f_components, num_entries])

    forces = tf.decode_raw(example['forces'], tf.float64)
    forces.set_shape([num_f_components])

    return ForcesExample(features=features,
                         energy=energy,
                         occurs=occurs,
                         weights=weights,
                         y_weight=y_weight,
                         forces=forces,
                         coef=coef,
                         indexing=indexing)

  else:
    return EnergyExample(features=features,
                         energy=energy,
                         occurs=occurs,
                         weights=weights,
                         y_weight=y_weight)


def read_and_decode(filename_queue, cnk, ck2, num_atom_types):
  """
  Read and decode a single example from the TFRecords file for training energies
  only.

  Args:
    filename_queue: an input queue.
    cnk: an `int` as the value of C(N,k).
    ck2: an `int` as the value of C(k,2).
    num_atom_types: an `int` as the number of atom types.

  Returns:
    example: a decoded `TFExample` from the TFRecord file.

  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return decode_protobuf(serialized_example,
                         cnk=cnk,
                         ck2=ck2,
                         num_atom_types=num_atom_types)


def inputs(train, batch_size=50, shuffle=True, dataset_name=None):
  """
  Read the input data for training energies only.

  Args:
    train: a `bool` selecting between the training (True) and validation
      (False) data.
    batch_size: an `int` as the number of examples per returned batch.
    shuffle: a `bool` indicating whether the batches shall be shuffled or not.
    dataset_name: a `str` as the name of the dataset.

  Returns:
    batch: a `Batch`.
      * features: a `float` tensor with shape [batch_size, 1, C(N,k), C(k,2)] in
        the range [0.0, 1.0].
      * energies: a `float` tensor with shape `[batch_size, ]`.
      * weights: a `float` tensor with shape `[batch_size, C(N,k)]`.
      * occurs: a `float` tensor with shape `[batch_size, num_atom_types]`.
      * y_weight: a `float` tensor with shape `[batch_size, ]`.

    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

  """

  filename, _ = get_filenames(train=train, dataset=dataset_name)
  filenames = [filename]

  configs = inputs_configs(train=train, dataset_name=dataset_name)
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
    example = read_and_decode(filename_queue,
                              cnk,
                              ck2,
                              num_atom_types=num_atom_types)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if not shuffle:
      batches = tf.train.batch(
        example,
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size
      )
    else:
      batches = tf.train.shuffle_batch(
        example,
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000
      )
    return batches


def inputs_configs(train=True, dataset_name=None):
  """
  Return the configs for inputs.

  Args:
    train: boolean indicating if one should return settings for training or
      validation.
    dataset_name: a `str` as the name of the dataset.

  Returns:
    configs: a `dict` of configs.

  """
  _, cfgfile = get_filenames(train=train, dataset=dataset_name)
  with open(cfgfile) as f:
    return dict(json.load(f))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  if FLAGS.periodic and (FLAGS.format != 'grendel'):
    tf.logging.error(
      "The xyz format must be `grendel` if `periodic` is True!")
  elif FLAGS.run_test:
    test_dataset_api()
  else:
    may_build_dataset(verbose=True)


def test_dataset_api():
  """
  A simple example demonstrating how to use tf.contrib.data.Dataset APIs.
  """
  dataset = tf.contrib.data.TFRecordDataset(["./binary/qm7-train.tfrecords"])
  dataset = dataset.map(
    partial(decode_protobuf,
            cnk=4495,
            ck2=3,
            num_atom_types=6,
            atomic_forces=False))
  dataset = tf.contrib.data.Dataset.zip((dataset, dataset))
  dataset = dataset.batch(5)

  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, dataset.output_types, dataset.output_shapes)
  ((feed_batch_y), (feed_batch_f)) = iterator.get_next()
  y_true_y = feed_batch_y[1]
  y_true_f = feed_batch_f[1]

  training_iterator = dataset.make_one_shot_iterator()
  y = tf.add(y_true_y, y_true_f, name="y2")

  with tf.Session() as sess:
    training_handle = sess.run(training_iterator.string_handle())
    print(sess.run([y_true_y, y_true_f, y],
                   feed_dict={handle: training_handle}))


if __name__ == "__main__":
  tf.app.run(main=main)
