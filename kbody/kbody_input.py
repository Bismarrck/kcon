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
from os.path import join, isfile, isdir
from os import makedirs
from sklearn.model_selection import train_test_split
from scipy.misc import comb
from collections import Counter
from functools import partial

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


tf.app.flags.DEFINE_string("binary_dir", "./binary",
                           """The directory for storing binary datasets.""")
tf.app.flags.DEFINE_string('dataset', 'TaB20opted',
                           """Define the dataset to use. This is also the name
                           of the xyz file to load.""")
tf.app.flags.DEFINE_string('format', 'xyz',
                           """The format of the xyz file.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """The total number of examples to use.""")
tf.app.flags.DEFINE_integer('num_atoms', 17,
                            """The number of atoms in each molecule.""")
tf.app.flags.DEFINE_integer('many_body_k', 3,
                            """The many-body-expansion order.""")
tf.app.flags.DEFINE_boolean('two_body', False,
                            """Include a standalone two-body term or not.""")
tf.app.flags.DEFINE_float('test_size', 0.2,
                          """The proportion of the dataset to include in the 
                          test split""")
tf.app.flags.DEFINE_integer("order", 1,
                            """The exponential order for normalizing 
                            distances.""")
tf.app.flags.DEFINE_boolean('use_fp64', False,
                            """Use double precision floats if True.""")
tf.app.flags.DEFINE_boolean('run_input_test', False,
                            """Run the input unit test if True.""")
tf.app.flags.DEFINE_float('unit', None,
                          """Override the default unit if this is not None.""")
tf.app.flags.DEFINE_boolean('periodic', False,
                            """The isomers are periodic structures.""")
tf.app.flags.DEFINE_float('weighted_loss', None,
                          """The kT (eV) for computing the weighted loss. """)

FLAGS = tf.app.flags.FLAGS

# Set the random state
SEED = 218

# Hartree to kcal/mol
hartree_to_kcal = 627.509

# Hartree to eV
hartree_to_ev = 27.211

# a.u to angstroms
au_to_angstrom = 0.52917721092


def exp_loss_weight_fn(x, x0=0.0, beta=1.0):
  """
  An exponential function for computing the weighted loss.

  I.e. \\(y = \e^{-\beta \cdot (x - x_0)}\\).
  """
  return np.float32(np.exp(-(x - x0) * beta))


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
    energy_patt = re.compile(r"Lattice=\"(.*)\".*"
                             r"energy=([\d.-]+)\s+pbc=\"(.*)\"")
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
  lattices = np.zeros((num_examples, 9))
  pbcs = np.zeros((num_examples, 3), dtype=bool)
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
          elif xyz_format.lower() == 'grendel':
            energies[i] = float(m.group(2)) * unit
            lattices[i] = [float(x) for x in m.group(1).split()]
            pbcs[i] = [True if x == "T" else False for x in m.group(3).split()]
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
    lattices = np.resize(lattices, (i, 9))
    pbcs = np.resize(pbcs, (i, 3))

  return array_of_species, energies, coords, forces, lattices, pbcs


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
  xyzfile = join("..", "datasets", "%s.xyz" % FLAGS.dataset)
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  # Read atomic symbols, DFT energies, atomic coordinates and atomic forces from
  # the xyz file.
  species, energies, coordinates, forces, lattices, pbcs = extract_xyz(
    xyzfile,
    num_examples=FLAGS.num_examples,
    num_atoms=FLAGS.num_atoms,
    verbose=verbose,
    xyz_format=FLAGS.format,
  )
  min_ener = energies.min()
  indices = list(range(len(coordinates)))

  # Transform the coordinates to input features and save these features in a
  # tfrecords file.
  many_body_k = min(5, FLAGS.many_body_k)

  # Split the energies and coordinates into two sets: a training set and a
  # testing set. The indices are used for post-analysis.
  (coords_train, coords_test,
   energies_train, energies_test,
   indices_train, indices_test,
   species_train, species_test,
   lattices_train, lattices_test,
   pbcs_train, pbcs_test) = train_test_split(
    coordinates,
    energies,
    indices,
    species,
    lattices,
    pbcs,
    test_size=FLAGS.test_size,
    random_state=SEED
  )

  # Determine the maximum occurance of each atom type.
  max_occurs = {}
  for symbols in species:
    c = Counter(symbols)
    for specie, times in c.items():
      max_occurs[specie] = max(max_occurs.get(specie, 0), times)

  # Setup the function for computing weighted losses.
  if FLAGS.weighted_loss is None:
    loss_weight_fn = None
  else:
    beta = 1.0 / FLAGS.weighted_loss
    loss_weight_fn = partial(exp_loss_weight_fn, x0=min_ener, beta=beta)

  # Use a `FixedLenMultiTransformer` to generate features because it will be
  # much easier if the all input samples are fixed-length.
  clf = kbody_transform.FixedLenMultiTransformer(
    max_occurs,
    many_body_k=many_body_k,
    periodic=FLAGS.periodic,
    order=FLAGS.order,
    two_body=FLAGS.two_body,
  )
  clf.transform_and_save(species_test, energies_test, coords_test,
                         test_file, indices_test, lattices_test, pbcs_test,
                         one_body_weights=False, loss_weight_fn=loss_weight_fn)
  clf.transform_and_save(species_train, energies_train, coords_train,
                         train_file, indices_train, lattices_train, pbcs_train,
                         one_body_weights=True, loss_weight_fn=loss_weight_fn)


def read_and_decode(filename_queue, cnk, ck2, nat):
  """
  Read and decode the mixed binary dataset file.

  Args:
    filename_queue: an input queue.
    cnk: a `int` as the value of C(N,k). 
    ck2: a `int` as the value of C(k,2). 
    nat: a `int` as the number of atom types.

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
      'occurs': tf.FixedLenFeature([], tf.string),
      'weights': tf.FixedLenFeature([], tf.string),
      'loss_weight': tf.FixedLenFeature([], tf.float32)
    })

  features = tf.decode_raw(example['features'], dtype)
  features.set_shape([cnk * ck2])
  features = tf.reshape(features, [1, cnk, ck2])

  energy = tf.decode_raw(example['energy'], tf.float64)
  energy.set_shape([1])
  energy = tf.squeeze(energy)

  occurs = tf.decode_raw(example['occurs'], tf.float32)
  occurs.set_shape([nat])
  occurs = tf.reshape(occurs, [1, 1, nat])

  weights = tf.decode_raw(example['weights'], tf.float32)
  weights.set_shape([cnk, ])
  weights = tf.reshape(weights, [1, cnk, 1])

  loss_weight = tf.cast(example['loss_weight'], tf.float32)

  return features, energy, occurs, weights, loss_weight


def inputs(train, batch_size=25, shuffle=True, dataset=None):
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
  cnk = configs["total_dim"]
  nat = configs["nat"]
  ck2 = comb(FLAGS.many_body_k, 2, exact=True)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      filenames,
    )

    # Even when reading in multiple threads, share the filename
    # queue.
    features, energies, occurs, weights, loss_weight = read_and_decode(
      filename_queue, cnk, ck2, nat
    )

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if not shuffle:
      batches = tf.train.batch(
        [features, energies, occurs, weights, loss_weight],
        batch_size=batch_size,
        capacity=1000 + 3 * batch_size
      )
    else:
      batches = tf.train.shuffle_batch(
        [features, energies, occurs, weights, loss_weight],
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
  if FLAGS.run_input_test:
    test_extract_mixed_xyz()
    test_build_dataset()
  else:
    if FLAGS.periodic and (FLAGS.format != 'grendel'):
      tf.logging.error(
        "The xyz format must be `grendel` if `periodic` is True!")
      exit(1)
    may_build_dataset(verbose=True)


if __name__ == "__main__":
  tf.app.run(main=main)
