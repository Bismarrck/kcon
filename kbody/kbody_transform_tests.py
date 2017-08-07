# coding=utf-8
"""
The unittests for all transformers in `kbody_transform`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import transformer
from ase import Atoms
from utils import get_atoms_from_kbody_term
from itertools import repeat, chain, combinations
from scipy.misc import comb
from sklearn.metrics import pairwise_distances
from collections import Counter

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

epsilon = 1e-6


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def get_species(occurs):
  species = [list(repeat(elem, k)) for elem, k in occurs.items()]
  return sorted(chain(*species))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def get_example(num_atoms):
  return np.repeat(np.atleast_2d(np.arange(num_atoms)).T, repeats=3, axis=1)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
class TransformerTest(tf.test.TestCase):

  def test_simple(self):
    coords = get_example(21)
    species = get_species({"Ta": 1, "B": 20})
    k_max = 4
    clf = transformer.Transformer(species, k_max=k_max)
    shape = clf.shape
    self.assertEqual(clf.ck2, comb(k_max, 2, exact=True))
    self.assertListEqual(clf.split_dims, [4845, 1140])
    self.assertListEqual(clf.kbody_sizes, clf.split_dims)
    self.assertAlmostEqual(clf.binary_weights.sum(),
                           float(shape[0]), delta=0.0001)

    features = clf.transform(Atoms(species, coords))
    self.assertTupleEqual(features.shape, (5985, 6))
    orders = np.argsort(features[0, :]).tolist()
    self.assertListEqual(orders, list(range(6)))
    orders = np.argsort(features[-1, [2, 4, 5]]).tolist()
    self.assertListEqual(orders, list(range(3)))
    orders = np.argsort(features[-1, [0, 1, 3]]).tolist()
    self.assertListEqual(orders, list(range(3)))

  def test_ghost(self):
    species = get_species({"C": 7, "H": 9, "N": 3, "X": 1})
    k_max = 3
    kbody_terms = sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, k_max)])))
    num_terms = len(kbody_terms)
    coords = np.array([
      [0.15625000, 1.42857141, 0.00000000],
      [0.51290443, 0.41976140, 0.00000000],
      [0.51292284, 1.93296960, 0.87365150],
      [0.51292284, 1.93296960, -0.87365150],
      [-0.91375000, 1.42858459, 0.00000000]
    ], dtype=np.float64)
    clf = transformer.Transformer(
      species=get_species({"C": 1, "H": 4, "X": 1}),
      k_max=k_max,
      kbody_terms=kbody_terms
    )

    features = clf.transform(Atoms(clf.species, coords))
    # 4CH + 6HH + 6CHH + 4HHH + (10 - 2) + (6 - 2) = 32
    self.assertTupleEqual(features.shape, (32, 3))

  def test_fixed_kbody_terms(self):
    species = get_species({"C": 7, "H": 9, "N": 3})
    k_max = 3
    kbody_terms = sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, k_max)])))
    num_terms = len(kbody_terms)

    coords = get_example(5)
    clf = transformer.Transformer(
      species=get_species({"C": 1, "H": 4}),
      k_max=k_max,
      kbody_terms=kbody_terms
    )
    shape = clf.shape

    self.assertEqual(len(clf.split_dims), num_terms)
    self.assertEqual(len(clf.kbody_terms), num_terms)

    features = clf.transform(Atoms(clf.species, coords))
    self.assertTupleEqual(features.shape, (18, 3))
    self.assertListEqual(clf.split_dims, [1, 1, 1, 6, 1, 1, 4, 1, 1, 1])
    self.assertAlmostEqual(np.sum(features[0:3, :]), 0.0, delta=epsilon)

    d12 = np.linalg.norm(coords[1, :] - coords[2, :])
    s12 = 0.64
    self.assertAlmostEqual(features[3, 2], np.exp(-d12 / s12), delta=epsilon)

    selection = clf.kbody_selections['CHH']
    self.assertEqual(len(selection), comb(4, 2, exact=True))
    self.assertListEqual(selection[0], [0, 1, 2])
    self.assertListEqual(selection[1], [0, 1, 3])
    self.assertListEqual(selection[2], [0, 1, 4])
    self.assertListEqual(selection[3], [0, 2, 3])
    self.assertListEqual(selection[4], [0, 2, 4])
    self.assertListEqual(selection[5], [0, 3, 4])

  def test_fixed_split_dims(self):
    k_max = 3
    occurs = {"C": 7, "H": 9, "O": 4}
    species = get_species(occurs)
    kbody_terms = sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, k_max)])))
    num_terms = len(kbody_terms)
    split_dims = []
    for kbody_term in kbody_terms:
      counter = Counter(get_atoms_from_kbody_term(kbody_term))
      dims = [comb(occurs[e], k, True) for e, k in counter.items()]
      split_dims.append(np.prod(dims))
    split_dims = [int(x) for x in split_dims]

    clf = transformer.Transformer(
      species=get_species({"C": 2, "H": 4}),
      k_max=k_max,
      kbody_terms=kbody_terms,
      split_dims=split_dims
    )
    self.assertListEqual(clf.kbody_sizes, [0, 4, 0, 12, 0, 0, 4, 0, 0, 0])
    self.assertAlmostEqual(clf.binary_weights.sum(), 20.0, delta=0.0001)

    coords = get_example(6)
    features = clf.transform(Atoms(clf.species, coords))
    offsets = [0] + np.cumsum(clf.split_dims).tolist()
    selections = clf.kbody_selections
    self.assertEqual(features.shape[0], comb(20, k_max, exact=True))

    ccc = features[offsets[0]: offsets[1], :]
    self.assertAlmostEqual(np.sum(ccc), 0.0, delta=epsilon)

    cch = features[offsets[1]: offsets[2], :]
    dists = pairwise_distances(coords[selections['CCH'][0]])
    dists = dists[[0, 0, 1], [1, 2, 2]]
    lmat = [1.5, 1.07, 1.07]
    vsum = np.exp(-dists / np.asarray(lmat)).sum()
    self.assertAlmostEqual(vsum, cch[0].sum(), delta=epsilon)

    cco = features[offsets[2]: offsets[3], :]
    self.assertAlmostEqual(np.sum(cco), 0.0, delta=epsilon)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
class MultiTransformerTest(tf.test.TestCase):
  pass


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
class FixedLenMultiTransformerTest(tf.test.TestCase):

  def test_main(self):
    max_occurs = {"C": 3, "H": 4, "O": 3}
    max_natoms = sum(max_occurs.values())
    k_max = 3
    clf = transformer.FixedLenMultiTransformer(
      max_occurs,
      k_max=k_max
    )

    species = get_species(max_occurs)
    coords = get_example(len(species))
    features, split_dims, _, _ = clf.transform(Atoms(species, coords))
    total_dim = comb(max_natoms, k_max, True)

    self.assertTupleEqual(features.shape, (total_dim, 3))
    self.assertEqual(len(split_dims), 10)

    species = get_species({"C": 1, "H": 4})
    coords = get_example(len(species))
    features, split_dims_, _, _ = clf.transform(Atoms(species, coords))
    self.assertListEqual(list(split_dims), list(split_dims_))
    self.assertTupleEqual(features.shape, (total_dim, 3))


if __name__ == "__main__":
  tf.test.main()
