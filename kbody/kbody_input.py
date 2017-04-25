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
import json
import kbody_transform
from os.path import join, isfile, isdir, dirname
from os import makedirs
from sklearn.model_selection import train_test_split
from scipy.misc import comb
from collections import Counter

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

flags = tf.app.flags

flags.DEFINE_string("binary_dir", "./binary",
                    """The directory for storing binary datasets.""")
flags.DEFINE_string('dataset', 'TaB20opted',
                    """Define the dataset to use. This is also the name
                    of the xyz file to load.""")
flags.DEFINE_string('format', 'xyz',
                    """The format of the xyz file.""")
flags.DEFINE_integer('num_examples', 5000,
                     """The total number of examples to use.""")
flags.DEFINE_integer('num_atoms', 17,
                     """The number of atoms in each molecule.""")
flags.DEFINE_integer('many_body_k', 4,
                     """The many-body-expansion order.""")
flags.DEFINE_float('test_size', 0.2,
                   """The proportion of the dataset to include in the test 
                   split""")
flags.DEFINE_integer("order", 1,
                     """The exponential order for normalizing distances.""")
flags.DEFINE_boolean('use_fp64', False,
                     """Use double precision floats if True.""")
flags.DEFINE_boolean('run_input_test', False,
                     """Run the input unit test if True.""")
flags.DEFINE_float('unit', None,
                   """Override the default unit if this is not None.""")

FLAGS = flags.FLAGS

# Set the random state
SEED = 218

# Hartree to kcal/mol
hartree_to_kcal = 627.509

# Hartree to eV
hartree_to_ev = 27.211

# a.u to angstroms
au_to_angstrom = 0.52917721092


def get_float_type(convert=False):
  """
  Return the data type of floats in this mini-project.
  """
  if convert:
    return tf.float32
  else:
    return np.float32


def _get_regex_patt_and_unit(xyz_format):
  """
  Return the corresponding regex patterns and the energy unit.
  
  Args:
    xyz_format: a `str` as the format of the file. 

  Returns:
    energy_patt: a regex pattern for parsing energies.
    string_patt: a regex pattern for parsing atomic symbols and coordinates.
    unit: a `float` transforming the energies to eV.
    parse_forces: a `bool` indicating whether the file supports parsing forces.

  """
  if xyz_format.lower() == 'grendel':
    energy_patt = re.compile(r".*energy=([\d.-]+).*")
    string_patt = re.compile(
      r"([A-Za-z]{1,2})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+"
      "\d+\s+\d.\d+\s+\d+\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")
    unit = 1.0
    parse_forces = True
  elif xyz_format.lower() == 'cp2k':
    energy_patt = re.compile(r"i\s=\s+\d+,\sE\s=\s+([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
    parse_forces = False
    unit = hartree_to_ev
  elif xyz_format.lower() == 'xyz':
    energy_patt = re.compile(r"([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
    unit = hartree_to_ev
    parse_forces = False
  elif xyz_format.lower() == 'extxyz':
    energy_patt = re.compile(r"i=(\d+).(\d+),\sE=([\d.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
    unit = hartree_to_ev
    parse_forces = False
  else:
    raise ValueError("The file format of %s is not supported!" % xyz_format)
  if FLAGS.unit is not None:
    unit = FLAGS.unit
  return energy_patt, string_patt, unit, parse_forces


def extract_xyz(filename, num_examples, num_atoms, xyz_format='xyz',
                verbose=True):
  """
  Extract atomic species, energies, coordiantes, and perhaps forces, from the 
  file.

  Args:
    filename: a `str` as the file to parse.
    num_examples: a `int` as the maximum number of examples to parse.
    num_atoms: a `int` as the number of atoms. If `mixed` is True, this should 
      be the maximum number of atoms in one configuration.
    xyz_format: a `str` representing the format of the given xyz file.
    verbose: a `bool` indicating whether we should log the parsing progress.

  Returns
    array_of_species: an array of `Array[str]` as the species of molecules.
    energies: `Array[N, ]`, a 1D array as the atomic energies.
    coordinates: `Array[N, ...]`, a 3D array as the atomic coordinates.
    forces: `Array[N, ...]`, a 3D array as the atomic forces.

  """

  dtype = get_float_type(convert=False)
  energies = np.zeros((num_examples,), dtype=np.float64)
  coords = np.zeros((num_examples, num_atoms, 3), dtype=dtype)
  array_of_species = []
  species = []
  stage = 0
  i = 0
  j = 0
  n = None
  ener_patt, xyz_patt, unit, parse_forces = _get_regex_patt_and_unit(xyz_format)

  if parse_forces:
    forces = np.zeros((num_examples, num_atoms, 3), dtype=dtype)
  else:
    forces = None

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
          if n > num_atoms:
            raise ValueError("The number of atoms %d from the file is larger "
                             "than the given maximum %d!" % (n, num_atoms))
          stage += 1
      elif stage == 1:
        m = ener_patt.search(l)
        if m:
          if xyz_format.lower() == 'extxyz':
            energies[i] = float(m.group(3)) * unit
          else:
            energies[i] = float(m.group(1)) * unit
          stage += 1
      elif stage == 2:
        m = xyz_patt.search(l)
        if m:
          coords[i, j, :] = [float(v) for v in m.groups()[1:4]]
          if parse_forces:
            forces[i, j, :] = [float(v) for v in m.groups()[4:7]]
          species.append(m.group(1))
          j += 1
          if j == n:
            array_of_species.append(species)
            species = []
            j = 0
            stage = 0
            i += 1
            if verbose and i % 1000 == 0:
              sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  if i < num_examples:
    array_of_species = np.asarray(array_of_species)
    energies = np.resize(energies, (i, ))
    coords = np.resize(coords, (i, num_atoms, 3))
    if forces is not None:
      forces = np.resize(forces, (i, num_atoms, 3))

  return array_of_species, energies, coords, forces


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
    num_examples=FLAGS.num_examples,
    num_atoms=FLAGS.num_atoms,
    verbose=verbose,
    xyz_format=FLAGS.format,
  )
  indices = list(range(len(coordinates)))

  # Transform the coordinates to input features and save these features in a
  # tfrecords file.
  many_body_k = min(5, FLAGS.many_body_k)

  # Split the energies and coordinates into two sets: a training set and a
  # testing set. The indices are used for post-analysis.
  (coords_train, coords_test,
   energies_train, energies_test,
   indices_train, indices_test,
   species_train, species_test) = train_test_split(
    coordinates,
    energies,
    indices,
    species,
    test_size=FLAGS.test_size,
    random_state=SEED
  )

  # Determine the maximum occurance of each atom type.
  max_occurs = {}
  for symbols in species:
    c = Counter(symbols)
    for specie, times in c.items():
      max_occurs[specie] = max(max_occurs.get(specie, 0), times)

  # Use a `FixedLenMultiTransformer` to generate features because it will be
  # much easier if the all input samples are fixed-length.
  clf = kbody_transform.FixedLenMultiTransformer(max_occurs, many_body_k)
  clf.transform_and_save(species_test, energies_test, coords_test,
                         test_file, indices_test)
  clf.transform_and_save(species_train, energies_train, coords_train,
                         train_file, indices_train)


def read_and_decode(filename_queue, cnk, ck2):
  """
  Read and decode the mixed binary dataset file.

  Args:
    filename_queue: an input queue.
    cnk: a `int` as the value of C(N,k). 
    ck2: a `int` as the value of C(k,2). 

  Returns:
    features: a 3D array of shape [1, cnk, ck2] as one input feature matrix.
    energy: a 1D array of shape [1,] as the target for the features.
    weights: a 1D array of shape [cnk,] as the weights of the k-body contribs.

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
      'weights': tf.FixedLenFeature([], tf.string),
    })

  features = tf.decode_raw(example['features'], dtype)
  features.set_shape([cnk * ck2])
  features = tf.reshape(features, [1, cnk, ck2])

  energy = tf.decode_raw(example['energy'], tf.float64)
  energy.set_shape([1])
  energy = tf.squeeze(energy)

  weights = tf.decode_raw(example['weights'], tf.float32)
  weights.set_shape([cnk, ])
  weights = tf.reshape(weights, [1, cnk, 1])

  return features, energy, weights


def inputs(train, batch_size=25, shuffle=True):
  """
  Reads mixed input data.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    shuffle: boolean indicating whether the batches shall be shuffled or not.

  Returns:
    A tuple (features, energies, weights), where:
    * features is a float tensor with shape [batch_size, 1, C(N,k), C(k,2)] in
      the range [0.0, 1.0].
    * energies is a float tensor with shape [batch_size, ].
    * weights is a float tensor with shape [batch_size, C(N,k)]

    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

  """

  filename, _ = get_filenames(train=train)
  filenames = [filename]

  settings = inputs_settings(train=train)
  cnk = settings["total_dim"]
  ck2 = comb(FLAGS.many_body_k, 2, exact=True)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      filenames,
    )

    # Even when reading in multiple threads, share the filename
    # queue.
    features, energies, weights = read_and_decode(filename_queue, cnk, ck2)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if not shuffle:
      batch_features, batch_energies, batch_weights = tf.train.batch(
        [features, energies, weights],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size
      )
    else:
      batch_features, batch_energies, batch_weights = tf.train.shuffle_batch(
        [features, energies, weights],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000
      )

    return batch_features, batch_energies, batch_weights


def inputs_settings(train=True):
  """
  Return the global settings for inputs.

  Args:
    train: boolean indicating if one should return settings for training or
      validation.

  Returns:
    settings: a dict of settings.

  """
  _, cfgfile = get_filenames(train=train)
  with open(cfgfile) as f:
    return dict(json.load(f))


def test_extract_mixed_xyz():
  """
  Test parsing the mixed xyz file `qm7.xyz`.
  """
  from scipy.io import loadmat

  mixed_file = join(dirname(__file__), "..", "datasets", "qm7.xyz")
  array_of_species, energies, coords, _ = extract_xyz(
    mixed_file,
    num_examples=100000,
    num_atoms=23,
  )
  qm7_mat = join(dirname(__file__), "..", "datasets", "qm7.mat")
  ar = loadmat(qm7_mat)
  kcal_to_hartree = 1.0 / 627.509474
  t = ar["T"].flatten() * kcal_to_hartree * hartree_to_ev
  r = np.multiply(ar["R"], au_to_angstrom)
  lefts = np.ones(len(array_of_species), dtype=bool)
  for i in range(len(array_of_species)):
    j = np.argmin(np.abs(energies[i] - t))
    if lefts[j]:
      n = len(array_of_species[i])
      d = np.linalg.norm(coords[i][:n] - r[j][:n])
      if d < 0.1:
        lefts[j] = False
  assert not np.all(lefts)


def test_build_dataset():
  """
  Test building the mixed QM7 dataset.
  """

  from sklearn.metrics import pairwise_distances

  xyzfile = join("..", "datasets", "qm7.xyz")
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  array_of_species, raw_energies, raw_coordinates, _ = extract_xyz(
    xyzfile,
    num_examples=8000,
    num_atoms=23,
    verbose=False,
  )
  indices = list(range(len(raw_coordinates)))

  (coords_train, coords_test,
   energies_train, energies_test,
   indices_train, indices_test,
   array_of_species_train, array_of_species_test) = train_test_split(
    raw_coordinates,
    raw_energies,
    indices,
    array_of_species,
    test_size=0.2,
    random_state=SEED
  )

  r_cc = 1.5
  many_body_k = 3
  max_occurs = {}
  for symbols in array_of_species:
    c = Counter(symbols)
    for specie, times in c.items():
      max_occurs[specie] = max(max_occurs.get(specie, 0), times)
  clf = kbody_transform.FixedLenMultiTransformer(max_occurs, many_body_k)

  train_file = join(FLAGS.binary_dir, "qm7-train.tfrecords")
  if not isfile(train_file):
    clf.transform_and_save(array_of_species_train, energies_train, coords_train,
                           train_file, indices=indices)

  with tf.Session() as sess:

    x_op, y_pred_op, w_op = inputs(train=True, shuffle=False, batch_size=5)
    tf.train.start_queue_runners(sess=sess)

    # --------
    # Features
    # --------
    features, energies, weights = sess.run([x_op, y_pred_op, w_op])
    species = array_of_species_train[1]
    n = len(species)
    coords = coords_train[1][:n]
    dists = pairwise_distances(coords[:3])
    v = np.sort(np.exp(-dists / r_cc)[[0, 0, 1], [1, 2, 2]])
    assert np.linalg.norm(v - features[1, 0, 0]) < 0.001
    assert np.abs(weights[1].flatten().sum() - comb(n, 3)) < 0.001

    time.sleep(5)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  if FLAGS.run_input_test:
    test_extract_mixed_xyz()
    test_build_dataset()
  else:
    may_build_dataset(verbose=True)


if __name__ == "__main__":
  tf.app.run(main=main)
