# coding=utf-8
"""
This script is used to transform atomic coordinates to input features and then
save them in tfrecords files.
"""

from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import sys
import json
from scipy.misc import comb
from itertools import combinations, product, repeat, chain
from sklearn.metrics import pairwise_distances
from collections import Counter
from os.path import basename, dirname, join, splitext, isfile
from os import remove
from tensorflow.python.training.training import Features, Example

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The pyykko radius of each element.
pyykko = {
  'Ac': 1.86, 'Ag': 1.28, 'Al': 1.26, 'Am': 1.66, 'Ar': 0.96, 'As': 1.21,
  'At': 1.47, 'Au': 1.24, 'B': 0.85, 'Ba': 1.96, 'Be': 1.02, 'Bh': 1.41,
  'Bi': 1.51, 'Bk': 1.68, 'Br': 1.14, 'C': 0.75, 'Ca': 1.71, 'Cd': 1.36,
  'Ce': 1.63, 'Cf': 1.68, 'Cl': 0.99, 'Cm': 1.66, 'Co': 1.11, 'Cr': 1.22,
  'Cs': 2.32, 'Cu': 1.12, 'Db': 1.49, 'Ds': 1.28, 'Dy': 1.67, 'Er': 1.65,
  'Es': 1.65, 'Eu': 1.68, 'F': 0.64, 'Fe': 1.16, 'Fm': 1.67, 'Fr': 2.23,
  'Ga': 1.24, 'Gd': 1.69, 'Ge': 1.21, 'H': 0.32, 'He': 0.46, 'Hf': 1.52,
  'Hg': 1.33, 'Ho': 1.66, 'Hs': 1.34, 'I': 1.33, 'In': 1.42, 'Ir': 1.22,
  'K': 1.96, 'Kr': 1.17, 'La': 1.8, 'Li': 1.33, 'Lu': 1.62, 'Md': 1.73,
  'Mg': 1.39, 'Mn': 1.19, 'Mo': 1.38, 'Mt': 1.29, 'N': 0.71, 'Na': 1.55,
  'Nb': 1.47, 'Nd': 1.74, 'Ne': 0.67, 'Ni': 1.1, 'No': 1.76, 'Np': 1.71,
  'O': 0.63, 'Os': 1.29, 'P': 1.11, 'Pa': 1.69, 'Pb': 1.44, 'Pd': 1.2,
  'Pm': 1.73, 'Po': 1.45, 'Pr': 1.76, 'Pt': 1.23, 'Pu': 1.72, 'Ra': 2.01,
  'Rb': 2.1, 'Re': 1.31, 'Rf': 1.57, 'Rh': 1.25, 'Rn': 1.42, 'Ru': 1.25,
  'S': 1.03, 'Sb': 1.4, 'Sc': 1.48, 'Se': 1.16, 'Sg': 1.43, 'Si': 1.16,
  'Sm': 1.72, 'Sn': 1.4, 'Sr': 1.85, 'Ta': 1.46, 'Tb': 1.68, 'Tc': 1.28,
  'Te': 1.36, 'Th': 1.75, 'Ti': 1.36, 'Tl': 1.44, 'Tm': 1.64, 'U': 1.7,
  'V': 1.34, 'W': 1.37, 'X': 0.32, 'Xe': 1.31, 'Y': 1.63, 'Yb': 1.7,
  'Zn': 1.18, 'Zr': 1.54
}


def _get_pyykko_bonds_matrix(species, factor=1.5, flatten=True):
  """
  Return the pyykko-bonds matrix given a list of atomic symbols.

  Args:
    species: List[str], a list of atomic symbols.
    factor: a float, the scaling factor.
    flatten: a bool. The bonds matrix will be flatten to a 1D array if True.

  Returns:
    bonds: the bonds matrix (or vector if `flatten` is True).

  """
  rr = np.asarray([pyykko[specie] for specie in species])[:, np.newaxis]
  lmat = np.multiply(factor, rr + rr.T)
  if flatten:
    return lmat.flatten()
  else:
    return lmat


def _gen_dist2inputs_mapping(species, kbody_terms):
  """
  Build the mapping from interatomic distances matrix to the [C(N,k), C(k,2)]
  feature matrix.

  Args:
    species: a list of str as the ordered atomic symbols.
    kbody_terms: a list of comma-separated elements string as the ordered k-body
      atomic symbol combinations.

  Returns:
    mapping: a dict
    selection: a dict

  """
  natoms = len(species)
  uniques = set(species)
  indices = {}
  for element in uniques:
    for i in range(natoms):
      if species[i] == element:
        indices[element] = indices.get(element, []) + [i]
  mapping = {}
  selections = {}
  for term in kbody_terms:
    elements = term.split(",")
    ck2 = comb(len(elements), 2, exact=True)
    c = Counter(elements)
    keys = sorted(c.keys())
    candidates = [[list(o) for o in combinations(indices[e], c[e])]
                  for e in keys]
    # All k-order combinations of elements
    pairs = [list(chain(*o)) for o in product(*candidates)]
    selections[term] = pairs
    cnk = len(pairs)
    mapping[term] = np.zeros((ck2, cnk), dtype=int)
    for i in range(cnk):
      for j, (vi, vj) in enumerate(combinations(pairs[i], 2)):
        mapping[term][j, i] = vi * natoms + vj
  return mapping, selections


def _gen_sorting_indices(orders):
  """
  Generate the soring indices.

  Args:
    orders: a List as the ordered many-body atomic symbol combiantions.

  Returns:
    indices: a dict of indices for sorting along the last axis of the input
      features.

  """
  indices = {}
  for order in orders:
    elements = list(sorted(order.split(",")))
    atom_pairs = list(combinations(elements, r=2))
    n = len(atom_pairs)
    counter = Counter(atom_pairs)
    if max(counter.values()) == 1:
      continue
    indices[order] = []
    for pair, times in counter.items():
      if times > 1:
        indices[order].append([i for i in range(n) if atom_pairs[i] == pair])
  return indices


def _exponential(d, s):
  """
  Do the exponential scaling on the given array `d`.

  Args:
    d: an `np.ndarray`.
    s: a float or an `np.ndarray` with the same shape of `d` as the scaling
      factor(s).

  Returns:
    ds: the scaled array.

  """
  return np.exp(-d / s)


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Transformer:
  """
  This class is used to transform atomic coordinates and energies to input
  features and training targets.
  """

  def __init__(self, species, many_body_k=4):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      many_body_k: a `int` as the maximum order for the many-body expansion.

    """
    orders = sorted(list(set(
      [",".join(sorted(c)) for c in combinations(species, many_body_k)])))
    mapping, selections = _gen_dist2inputs_mapping(species, orders)
    offsets = [0]
    cnk = 0
    ck2 = None
    for order in orders:
      cnk += mapping[order].shape[1]
      if not ck2:
        ck2 = mapping[order].shape[0]
      offsets.append(cnk)
    self._many_body_k = many_body_k
    self._orders = orders
    self._species = species
    self._kbody_offsets = offsets
    self._dist2inputs_mapping = mapping
    self._selections = selections
    self._kbody_sizes = np.diff(offsets)
    self._cnk = cnk
    self._ck2 = ck2
    self._sorting_indices = _gen_sorting_indices(orders)
    self._lmat = _get_pyykko_bonds_matrix(species)

  @property
  def cnk(self):
    """
    Return the value of C(N,k) for this transformer.
    """
    return self._cnk

  @property
  def ck2(self):
    """
    Return the value of C(k,2) for this transformer.
    """
    return self._ck2

  @property
  def many_body_k(self):
    """
    Return the maximum order for the many-body expansion.
    """
    return self._many_body_k

  def transform(self, coordinates, energies):
    """
    Transform the given atomic coordinates and energies to input features and
    training targets and return them as numpy arrays.

    Args:
      coordinates: a 3D array as the atomic coordinates of sturctures.
      energies: a 1D array as the desired energies.

    Returns:
      features: a 4D array as the transformed input features.
      targets: a 1D array as the training targets (actually the negative of the
        input energies.)

    """
    num_examples = len(coordinates)
    samples = np.zeros((num_examples, self.cnk, self.ck2), dtype=np.float32)
    orders = self._orders
    mapping = self._dist2inputs_mapping
    offsets = self._kbody_offsets

    for i in range(num_examples):
      dists = pairwise_distances(coordinates[i]).flatten()
      rr = _exponential(dists, self._lmat)
      samples[i].fill(0.0)
      for j, order in enumerate(orders):
        for k in range(self.ck2):
          samples[i, offsets[j]: offsets[j + 1], k] = rr[mapping[order][k]]

      for j, order in enumerate(orders):
        for ix in self._sorting_indices.get(order, []):
          z = sample[i, offsets[j]: offsets[j + 1], ix]
          z.sort(axis=1)
          samples[i, offsets[j]: offsets[j + 1], ix] = z

    return samples, np.negative(energies)

  def _transform_and_save(self, coordinates, energies, filename, verbose):
    """
    The main function for transforming coordinates to input features.
    """
    with tf.python_io.TFRecordWriter(filename) as writer:

      if verbose:
        print("Start transforming %s ... " % filename)

      num_examples = len(coordinates)
      sample = np.zeros((self.cnk, self.ck2), dtype=np.float32)
      orders = self._orders
      mapping = self._dist2inputs_mapping
      offsets = self._kbody_offsets

      for i in range(num_examples):
        dists = pairwise_distances(coordinates[i]).flatten()
        rr = _exponential(dists, self._lmat)
        sample.fill(0.0)
        for j, order in enumerate(orders):
          for k in range(self.ck2):
            sample[offsets[j]: offsets[j + 1], k] = rr[mapping[order][k]]

        for j, order in enumerate(orders):
          for ix in self._sorting_indices.get(order, []):
            z = sample[offsets[j]: offsets[j + 1], ix]
            z.sort(axis=1)
            sample[offsets[j]: offsets[j + 1], ix] = z

        x = _bytes_feature(sample.tostring())
        y = _bytes_feature(np.atleast_2d(-1.0 * energies[i]).tostring())
        example = Example(
          features=Features(feature={'energy': y, 'features': x}))
        writer.write(example.SerializeToString())

        if verbose and i % 100 == 0:
          sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))

      if verbose:
        print("")
        print("Transforming %s finished!" % filename)

  def transform_and_save(self, coordinates, energies, filename, verbose=True,
                         indices=None):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      coordinates: a 3D array as the atomic coordinates of sturctures.
      energies: a 1D array as the desired energies.
      filename: a `str` as the file to save examples.
      verbose: boolean indicating whether.
      indices: a `List[int]` as the indices of each given example. This is an
        optional argument.

    """
    try:
      self._transform_and_save(coordinates, energies, filename, verbose)
    except Exception as excp:
      if isfile(filename):
        remove(filename)
      raise excp
    else:
      self._save_auxiliary_for_file(filename, indices)

  def _save_auxiliary_for_file(self, filename, indices=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      indices: a `List[int]` as the indices of each given example.

    """
    name = splitext(basename(filename))[0]
    workdir = dirname(filename)
    cfgfile = join(workdir, "{}.json".format(name))
    if indices is not None:
      if isinstance(indices, np.ndarray):
        indices = indices.tolist()
    else:
      indices = []

    with open(cfgfile, "w+") as f:
      json.dump({
        "kbody_offsets": self._kbody_offsets,
        "kbody_terms": self._orders,
        "kbody_selections": self._selections,
        "kbody_term_sizes": self._kbody_sizes.tolist(),
        "inverse_indices": list([int(i) for i in indices])
      }, f)


def _test_map_indices():
  species = ["Li"] + list(repeat("B", 6))
  orders = list(set([",".join(sorted(c)) for c in combinations(species, 4)]))
  mapping = _gen_dist2inputs_mapping(species, orders)
  assert mapping['B,B,B,Li'][0, 0] == 9


def _test_split_tensor():

  raw_inputs = np.arange(36, dtype=np.float32).reshape((1, 1, 6, 6))
  raw_offsets = [4, 2]

  inputs = tf.constant(raw_inputs)
  partitions = tf.split(inputs, raw_offsets, axis=2)

  with tf.Session() as sess:

    values = sess.run(partitions)

    assert len(values) == 2
    assert np.linalg.norm(values[0] - raw_inputs[:, :, 0:4, :]) == 0.0
    assert np.linalg.norm(values[0] - raw_inputs[:, :, 4:6, :]) == 0.0


if __name__ == "__main__":
  _test_map_indices()
  _test_split_tensor()
