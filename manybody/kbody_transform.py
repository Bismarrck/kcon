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
from os.path import basename, dirname, join, splitext

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


def get_pyykko_bonds_matrix(species, factor=1.5, flatten=True):
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


def map_indices(species, orders):
  """
  Build the mapping from interatomic distances matrix to the [C(N,k), C(k,2)]
  feature matrix.

  Args:
    species: a list of str as the ordered atomic symbols.
    orders: a list of comma-separated elements string as the ordered many-body
      atomic symbol combinations.

  Returns:
    mapping: a dict

  """
  natoms = len(species)
  uniques = set(species)
  indices = {}
  for element in uniques:
    for i in range(natoms):
      if species[i] == element:
        indices[element] = indices.get(element, []) + [i]
  mapping = {}
  for order in orders:
    elements = order.split(",")
    ck2 = comb(len(elements), 2, exact=True)
    c = Counter(elements)
    keys = sorted(c.keys())
    candidates = [[list(o) for o in combinations(indices[e], c[e])]
                  for e in keys]
    # All k-order combinations of elements
    pairs = [list(chain(*o)) for o in product(*candidates)]
    cnk = len(pairs)
    mapping[order] = np.zeros((ck2, cnk), dtype=int)
    for i in range(cnk):
      for j, (vi, vj) in enumerate(combinations(pairs[i], 2)):
        mapping[order][j, i] = vi * natoms + vj
  return mapping


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


def transform_and_save(coordinates, energies, species, orders, filename):
  """
  Transform the given atomic coordinates to input features and save them to
  tfrecords files using `tf.TFRecordWriter`.

  Args:
    coordinates: a 3D array as the atomic coordinates of sturctures.
    energies: a 1D array as the desired DFT energies.
    species: a List as the ordered atomic symboles.
    orders: a List as the ordered many-body atomic symbol combiantions.
    filename: a str, the file to save examples.

  """

  mapping = map_indices(species, orders)
  writer = tf.python_io.TFRecordWriter(filename)
  offsets = [0]
  cnk = 0
  ck2 = None
  for order in orders:
    cnk += mapping[order].shape[1]
    if not ck2:
      ck2 = mapping[order].shape[0]
    offsets.append(cnk)
  offsets.append(-1)
  offsets = np.asarray(offsets, dtype=np.int64)
  sizes = np.diff(offsets[:-1])

  sample = np.zeros((cnk, ck2), dtype=coordinates.dtype)
  num_traj = len(coordinates)
  lmat = get_pyykko_bonds_matrix(species, flatten=True)

  print("k-body terms: ")
  for order in orders:
    print("%11s : %d" % (order, mapping[order].shape[1]))
  print("")

  print("Start transforming %s ... " % filename)

  for i in range(num_traj):
    dists = pairwise_distances(coordinates[i]).flatten()
    rr = _exponential(dists, lmat)
    sample.fill(0.0)
    for j, order in enumerate(orders):
      for k in range(ck2):
        sample[offsets[j]: offsets[j + 1], k] = rr[mapping[order][k]]
    x = sample.tostring()
    y = np.atleast_2d(energies[i]).tostring()
    example = tf.train.Example(
      features=tf.train.Features(
        feature={'energy': _bytes_feature(y),
                 'features': _bytes_feature(x)}))
    writer.write(example.SerializeToString())

    if i % 100 == 0:
      sys.stdout.write("\rProgress: %7d / %7d" % (i, num_traj))

  print("")
  print("Transforming %s finished!" % filename)
  writer.close()

  cfgfile = join(
    dirname(filename),
    "%s.json" % basename(splitext(filename)[0])
  )
  print("Save configs to %s" % cfgfile)
  with open(cfgfile, "w+") as f:
    json.dump({"kbody_term_sizes": sizes.tolist(),
               "kbody_terms": orders}, f)
  print("")


def _test_map_indices():
  species = ["Li"] + list(repeat("B", 6))
  orders = list(set([",".join(sorted(c)) for c in combinations(species, 4)]))
  mapping = map_indices(species, orders)
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
