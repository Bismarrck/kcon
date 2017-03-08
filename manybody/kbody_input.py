# coding=utf-8
"""
This script is used to building training and validation datasets.
"""

from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import re
import time
import sys
import itertools
import json
from kbody_transform import transform_and_save
from os.path import join, isfile, isdir
from os import makedirs
from sklearn.model_selection import train_test_split
from scipy.misc import comb

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_examples", 2500,
                            """The total number of examples to use.""")
tf.app.flags.DEFINE_integer("num_atoms", 21,
                            """The number of atoms in each molecule.""")
tf.app.flags.DEFINE_integer("many_body_k", 4,
                            """The many-body-expansion order.""")
tf.app.flags.DEFINE_boolean("use_fp64", False,
                            """Use double precision floats if True.""")
tf.app.flags.DEFINE_boolean("parse_forces", False,
                            """Parse forces from the xyz file if True.""")
tf.app.flags.DEFINE_string("binary_dir", "./binary",
                           """The directory for storing binary datasets.""")
tf.app.flags.DEFINE_string("dataset", "TaB20opted",
                           """Define the dataset to use. This is also the name
                           of the xyz file to load.""")
tf.app.flags.DEFINE_string("format", "xyz",
                           """The format of the xyz file.""")
tf.app.flags.DEFINE_boolean("run_input_test", False,
                            """Run the input unit test if True.""")

# Set the random state
SEED = 218

# Hartree to kcal/mol
hartree_to_kcal = 627.509

# Hartree to eV
hartree_to_ev = 27.211


def get_float_type(convert=False):
  """
  Return the data type of floats in this mini-project.
  """
  if FLAGS.use_fp64:
    if convert:
      return tf.float64
    else:
      return np.float64
  else:
    if convert:
      return tf.float32
    else:
      return np.float32


def extract_xyz(filename, verbose=True, xyz_format='xyz', unit=1.0):
  """
  Extract atomic symbols, DFT energies and atomic coordiantes (and forces) from
  the file.

  Args:
    filename: a `str`, the file to parse.
    verbose: a `bool`.
    xyz_format: a `str` representing the format of the given xyz file.
    unit: a float represents the scaling of the energies.

  Returns
    species: `List[str]`, a list of the atomic symbols.
    energies: `Array[N, ]`, a 1D array as the atomic energies.
    coordinates: `Array[N, ...]`, a 3D array as the atomic coordinates.
    forces: `Array[N, ...]`, a 3D array as the atomic forces.

  """

  dtype = get_float_type(convert=False)
  num_examples = FLAGS.num_examples
  num_atoms = FLAGS.num_atoms
  energies = np.zeros((num_examples,), dtype=dtype)
  coordinates = np.zeros((num_examples, num_atoms, 3), dtype=dtype)

  if FLAGS.parse_forces:
    forces = np.zeros((num_examples, num_atoms, 3), dtype=dtype)
  else:
    forces = None
  species = []
  parse_species = True
  parse_forces = False
  stage = 0
  i = 0
  j = 0

  if xyz_format.lower() == 'grendel':
    energy_patt = re.compile(r".*energy=([\d.-]+).*")
    string_patt = re.compile(
      r"([A-Za-z]{1,2})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+"
      "\d+\s+\d.\d+\s+\d+\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")
    parse_forces = True
  elif xyz_format.lower() == 'cp2k':
    energy_patt = re.compile(r"i\s=\s+\d+,\sE\s=\s+([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
  elif xyz_format.lower() == 'xyz':
    energy_patt = re.compile(r"([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
  else:
    raise ValueError("The file format of %s is not supported!" % xyz_format)

  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")
  with open(filename) as f:
    for line in f:
      if i == num_examples:
        break
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          n = int(l)
          if n != num_atoms:
            raise ValueError("The parsed size %d != NUM_SITES" % n)
          stage += 1
      elif stage == 1:
        m = energy_patt.search(l)
        if m:
          energies[i] = float(m.group(1)) * unit
          stage += 1
      elif stage == 2:
        m = string_patt.search(l)
        if m:
          coordinates[i, j, :] = float(m.group(2)), float(m.group(3)), float(
            m.group(4))
          if parse_forces:
            forces[i, j, :] = float(m.group(5)), float(m.group(6)), float(
              m.group(7))
          if parse_species:
            species.append(m.group(1))
            if len(species) == num_atoms:
              species = np.asarray(species, dtype=object)
              parse_species = False
          j += 1
          if j == num_atoms:
            j = 0
            stage = 0
            i += 1
            if verbose and i % 1000 == 0:
              sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  return species, energies, coordinates, forces


def get_filenames(train=True):
  """
  Return the binary data file and the config file.

  Args:
    train: boolean indicating whether the training data file or the testing file
      should be returned.

  Returns:
    (binfile, jsonfile): the binary file to read data and the json file to read
      configs..

  """

  binary_dir = FLAGS.binary_dir
  if not isdir(binary_dir):
    makedirs(binary_dir)

  fname = FLAGS.dataset
  records = {"train": (join(binary_dir, "%s-train.tfrecords" % fname),
                       join(binary_dir, "%s-train.json" % fname)),
             "test": (join(binary_dir, "%s-test.tfrecords" % fname),
                      join(binary_dir, "%s-test.json" % fname))}

  if train:
    return records['train']
  else:
    return records['test']


def may_build_dataset(verbose=True):
  """
  Build the dataset if needed.

  Args:
    verbose: boolean indicating whether the building progress shall be printed
      or not.

  """
  train_file, _ = get_filenames(train=True)
  test_file, _ = get_filenames(train=False)

  # Check if the xyz file is accessible.
  xyzfile = join("..", "datasets", "%s.xyz" % FLAGS.dataset)
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  # Read atomic symbols, DFT energies, atomic coordinates and atomic forces from
  # the xyz file.
  species, energies, coordinates, forces = extract_xyz(
    xyzfile,
    verbose=verbose,
    xyz_format=FLAGS.format,
    unit=hartree_to_ev
  )

  # Determine the unique atomic symbol combinations.
  k = max(FLAGS.many_body_k, len(set(species)))
  orders = sorted(list(set(
    [",".join(sorted(c)) for c in itertools.combinations(species, k)])))

  # Split the energies and coordinates into two sets: a training set and a
  # testing set.
  coords_train, coords_test, energies_train, energies_test = train_test_split(
    coordinates,
    energies,
    test_size=0.2,
    random_state=SEED
  )

  # Transform the coordinates to input features and save these features in a
  # tfrecords file.
  transform_and_save(coords_test, energies_test, species, orders, test_file)
  transform_and_save(coords_train, energies_train,  species, orders, train_file)


height = comb(FLAGS.num_atoms, FLAGS.many_body_k, exact=True)
depth = comb(FLAGS.many_body_k, 2, exact=True)


def read_and_decode(filename_queue):
  """
  Read and decode the binary dataset file.

  Args:
    filename_queue: an input queue.

  """

  dtype = get_float_type(convert=True)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  example = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
      'features': tf.FixedLenFeature([], tf.string),
      'energy': tf.FixedLenFeature([], tf.string),
    })

  features = tf.decode_raw(example['features'], dtype)
  features.set_shape([height * depth])
  features = tf.reshape(features, [1, height, depth])

  energy = tf.decode_raw(example['energy'], dtype)
  energy.set_shape([1])
  energy = tf.squeeze(energy)

  return features, energy


def inputs(train, batch_size, num_epochs, shuffle=True):
  """
  Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    shuffle: boolean indicating whether the batches shall be shuffled or not.

  Returns:
    A tuple (features, energies, offsets), where:
    * features is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    * offsets

    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

  """
  if not num_epochs:
    num_epochs = None

  filename, cfgfile = get_filenames(train=train)

  with open(cfgfile) as f:
    offsets = dict(json.load(f))["kbody_term_sizes"]

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename],
      num_epochs=num_epochs
    )

    # Even when reading in multiple threads, share the filename
    # queue.
    features, energies = read_and_decode(filename_queue,)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if shuffle:
      batch_features, batch_energies = tf.train.shuffle_batch(
        [features, energies], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000
      )
    else:
      batch_features, batch_energies = tf.train.batch(
        [features, energies], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size
      )
    return batch_features, batch_energies, offsets


def test(*args):
  """
  This is the unit test of this module. The file `TaB2Oopted.xyz` is used.
  """
  from sklearn.metrics import pairwise_distances

  xyzfile = join("..", "datasets", "TaB20opted.xyz")
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  species, energies, coordinates, forces = extract_xyz(
    xyzfile,
    verbose=False,
    xyz_format=FLAGS.format,
    unit=hartree_to_ev
  )

  coords_train, _, energies_train, _ = train_test_split(
    coordinates,
    energies,
    test_size=0.2,
    random_state=SEED
  )

  dists = pairwise_distances(coords_train[0])
  pyykko_bb = 1.7 * 1.5
  pyykko_tb = 2.31 * 1.5

  train_file = join(FLAGS.binary_dir, "TaB20opted-train.tfrecords")
  if not isfile(train_file):
    orders = sorted(list(set(
      [",".join(sorted(c)) for c in itertools.combinations(species, 4)])))
    transform_and_save(coords_train, energies_train, species, orders,
                       train_file)

  with tf.Session() as sess:
    features, energies, offsets = inputs(
      train=True, batch_size=1, num_epochs=None, shuffle=False
    )
    tf.train.start_queue_runners(sess=sess)

    try:
      # The shape of `values` is [1, 1, 5985, 6].
      # The first part, [:, :, :4845, 6] represents the 4-body term of B-B-B-B.
      # The indices of Borons are from 1 to 20, so the first element should be
      # the scaled distance between B_1 and B_2.
      values = sess.run(features)

      r = values[0, 0, 0, 0]
      y = np.exp(-np.linalg.norm(dists[1, 2]) / pyykko_bb)
      assert np.linalg.norm(r - y) < 0.00001

      # The second part, [:, :, 4845:5985, 6], represents the 4-body term of
      # Ta-B-B-B. The index of Ta is 0, so the last element should be the scaled
      # distance of Ta_0 and B_20.
      r = values[0, 0, -1, -1]
      y = np.exp(-np.linalg.norm(dists[0, 20]) / pyykko_tb)
      assert np.linalg.norm(r - y) < 0.00001

    except AssertionError:
      print("The tests are failed!")
    else:
      print("The tests are passed!")


if __name__ == "__main__":
  if tf.app.flags.FLAGS.run_input_test:
    tf.app.run(test)
  else:
    tf.app.run(may_build_dataset, argv=(True, ))
