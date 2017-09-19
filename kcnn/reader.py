# coding=utf-8
"""
This script is used to building training and validation datasets.
"""

from __future__ import print_function, absolute_import

import json
import tensorflow as tf
from collections import namedtuple
from functools import partial
from os import makedirs
from os.path import join, isdir
from tensorflow.contrib.data.python.ops.dataset_ops import TFRecordDataset
from constants import SEED

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


tf.app.flags.DEFINE_string("binary_dir", "./binary",
                           """The directory for storing binary datasets.""")
tf.app.flags.DEFINE_string('dataset', 'C9H7N.PBE',
                           """Define the dataset to use. This is also the name
                           of the xyz file to load.""")
tf.app.flags.DEFINE_integer('k_max', 3,
                            """The maximum k under the many-body-expansion 
                            scheme.""")
tf.app.flags.DEFINE_boolean('include_all_k', True,
                            """Include all k-body terms from k = 1 to k_max.""")
tf.app.flags.DEFINE_boolean('run_input_test', False,
                            """Run the input unit test if True.""")
tf.app.flags.DEFINE_float('weighted_loss', None,
                          """The kT (eV) for computing the weighted loss. """)
tf.app.flags.DEFINE_boolean('forces', False,
                            """Set this to True to enable atomic forces.""")

FLAGS = tf.app.flags.FLAGS


def get_filenames(train=True, dataset_name=None):
  """
  Return the binary data file and the config file.

  Args:
    train: boolean indicating whether the training data file or the testing file
      should be returned.
    dataset_name: a `str` as the name of the dataset.

  Returns:
    (binfile, jsonfile): the binary file to read data and the json file to read
      configs..

  """
  binary_dir = FLAGS.binary_dir
  if not isdir(binary_dir):
    makedirs(binary_dir)

  name = dataset_name or FLAGS.dataset
  records = {"train": (join(binary_dir, "{}-train.tfrecords".format(name)),
                       join(binary_dir, "{}-train.json".format(name))),
             "test": (join(binary_dir, "{}-test.tfrecords".format(name)),
                      join(binary_dir, "{}-test.json".format(name)))}

  if train:
    return records['train']
  else:
    return records['test']


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


def y_inputs(train, batch_size=50, shuffle=True, dataset_name=None):
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

  filename, _ = get_filenames(train=train, dataset_name=dataset_name)
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
  _, json_file = get_filenames(train=train, dataset_name=dataset_name)
  with open(json_file) as f:
    return dict(json.load(f))


def yf_inputs(dataset_name, for_training=True, batch_size=50, shuffle=True):
  """
  Provide inputs for kCON energy & forces models.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` selecting between the training (True) and validation
      (False) data.
    batch_size: an `int` as the number of examples per returned batch.
    shuffle: a `bool` indicating whether the batches shall be shuffled or not.

  """
  tfrecods_file, _ = get_filenames(
    train=for_training,
    dataset_name=dataset_name
  )

  configs = inputs_configs(train=for_training, dataset_name=dataset_name)
  shape = configs["shape"]
  cnk = shape[0]
  ck2 = shape[1]
  num_atom_types = configs["num_atom_types"]
  atomic_forces = configs["atomic_forces_enabled"]
  if not atomic_forces:
    raise ValueError()
  num_f_components, num_entries = configs["indexing_shape"]

  def _build_dataset(filename):
    """
    A helper function for building a `tf.contrib.data.Dataset`.

    Args:
      filename: a `str` as the TFRecords file to read.

    Returns:
      feed_batch_: a tuple of tensors (see `ForcesExample`).
      handle_: a placeholder tensor that should be filled.
      dataset_iterator_: a iterator tensor that should be feeded to `handle`.

    """
    dataset = TFRecordDataset([filename]).map(
      partial(decode_protobuf,
              cnk=cnk,
              ck2=ck2,
              num_atom_types=num_atom_types,
              atomic_forces=atomic_forces,
              num_f_components=num_f_components,
              num_entries=num_entries)
    )
    dataset = dataset.repeat()
    if shuffle:
      dataset = dataset.shuffle(buffer_size=100000, seed=SEED)
    dataset = dataset.batch(batch_size)

    handle_ = tf.placeholder(tf.string, shape=[], name="handle")
    iterator = tf.contrib.data.Iterator.from_string_handle(
      handle_,
      dataset.output_types,
      dataset.output_shapes
    )
    feed_batch_ = iterator.get_next()
    dataset_iterator_ = dataset.make_initializable_iterator()
    return feed_batch_, handle_, dataset_iterator_

  if for_training:
    handles = []
    feed_batches = []
    dataset_iterators = []
    with tf.name_scope("inputs"):
      for scope in ("energy", "forces"):
        with tf.name_scope(scope):
          feed_batch, handle, dataset_iterator = _build_dataset(tfrecods_file)
          handles.append(handle)
          dataset_iterators.append(dataset_iterator)
          feed_batches.append(feed_batch)
      return feed_batches, handles, dataset_iterators

  else:
    return _build_dataset(tfrecods_file)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  test_dataset_api()


def test_dataset_api():
  """
  A simple example demonstrating how to use tf.contrib.data.Dataset APIs.
  """

  datasets = []
  handles = []
  feed_batches = []
  dataset_iterators = []

  for i in range(2):
    dataset = tf.contrib.data.TFRecordDataset(["./binary/qm7-train.tfrecords"])
    dataset = dataset.map(
      partial(decode_protobuf,
              cnk=4495,
              ck2=3,
              num_atom_types=6,
              atomic_forces=False))
    dataset = dataset.shuffle(buffer_size=100000, seed=1)
    dataset = dataset.batch(5)

    handle = tf.placeholder(tf.string, shape=[], name="handle{}".format(i))
    iterator = tf.contrib.data.Iterator.from_string_handle(
      handle,
      dataset.output_types,
      dataset.output_shapes
    )
    feed_batch = iterator.get_next()
    dataset_iterator = dataset.make_initializable_iterator()

    datasets.append(dataset)
    handles.append(handle)
    feed_batches.append(feed_batch)
    dataset_iterators.append(dataset_iterator)

  y_true_y = feed_batches[0][1]
  y_true_f = feed_batches[1][1]

  with tf.Session() as sess:

    training_handles = []

    for i in range(2):
      sess.run(dataset_iterators[i].initializer)
      training_handle = sess.run(dataset_iterators[i].string_handle())
      training_handles.append(training_handle)

    for i in range(10):
      print("Round = {}".format(i))
      print(sess.run(y_true_y,
                     feed_dict={handles[0]: training_handles[0]}))
      print(sess.run(y_true_f,
                     feed_dict={handles[1]: training_handles[1]}))


if __name__ == "__main__":
  tf.app.run(main=main)
