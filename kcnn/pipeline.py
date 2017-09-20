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


def get_configs(for_training=True, dataset_name=None):
  """
  Return the configs for inputs.

  Args:
    for_training: boolean indicating if one should return settings for training
      or validation.
    dataset_name: a `str` as the name of the dataset.

  Returns:
    configs: a `dict` of configs.

  """
  _, json_file = get_filenames(train=for_training, dataset_name=dataset_name)
  with open(json_file) as f:
    return dict(json.load(f))


def get_dataset_size(dataset_name, for_training=True):
  """
  Return the total number of examples in the given datatset.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` indicating whether we should return the size of the
      training set or validation set.

  Returns:
    num_examples: an `int` as the number of examples in the dataset.

  """
  return len(get_configs(for_training, dataset_name)['lookup_indices'])


def next_batch(dataset_name, for_training=True, batch_size=50, num_epochs=None,
               shuffle=True):
  """
  Provide batched inputs for kCON.

  Args:
    dataset_name: a `str` as the name of the dataset.
    for_training: a `bool` selecting between the training (True) and validation
      (False) data.
    batch_size: an `int` as the number of examples per batch.
    num_epochs: an `int` as the maximum number of epochs to run.
    shuffle: a `bool` indicating whether the batches shall be shuffled or not.

  Returns:
    next_batch: a tuple of Tensors.

  """

  with tf.device('/cpu:0'):
    tfrecods_file, _ = get_filenames(
      train=for_training,
      dataset_name=dataset_name
    )

    configs = get_configs(for_training=for_training, dataset_name=dataset_name)
    shape = configs["shape"]
    cnk = shape[0]
    ck2 = shape[1]
    num_atom_types = configs["num_atom_types"]
    atomic_forces = configs["atomic_forces_enabled"]
    if atomic_forces:
      num_f_components, num_entries = configs["indexing_shape"]
    else:
      num_f_components, num_entries = None, None

    # Initialize a basic dataset
    dataset = TFRecordDataset([tfrecods_file]).map(
      partial(decode_protobuf,
              cnk=cnk,
              ck2=ck2,
              num_atom_types=num_atom_types,
              atomic_forces=atomic_forces,
              num_f_components=num_f_components,
              num_entries=num_entries)
    )

    # Repeat the dataset
    dataset = dataset.repeat(count=num_epochs)

    # Shuffle it if needed
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000, seed=SEED)

    # Setup the batch
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
