# coding=utf-8
"""
The unittests for all transformers in `kbody_transform`.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody_transform
from itertools import repeat, chain, combinations
from scipy.misc import comb
from sklearn.metrics import pairwise_distances
from collections import Counter
from unittest import skip

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

epsilon = 1e-6


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def get_species(occurs):
  species = [list(repeat(elem, k)) for elem, k in occurs.items()]
  return sorted(chain(*species))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def get_example(num_atoms):
  return np.repeat(np.atleast_2d(
    np.arange(num_atoms)).T, repeats=3, axis=1).reshape((1, num_atoms, 3))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
@skip
class TransformerTest(tf.test.TestCase):

  @skip
  def test_simple(self):
    coords = get_example(21)
    species = get_species({"Ta": 1, "B": 20})
    many_body_k = 4
    clf = kbody_transform.Transformer(species, many_body_k=many_body_k)
    self.assertEqual(clf.ck2, comb(many_body_k, 2, exact=True))
    self.assertEqual(clf.cnk, comb(len(species), many_body_k, exact=True))
    self.assertListEqual(clf.split_dims, [4845, 1140])

    features, _ = clf.transform(coords, [0.0])
    self.assertTupleEqual(features.shape, (1, 5985, 6))
    orders = np.argsort(features[0, 0, :]).tolist()
    self.assertListEqual(orders, list(range(6)))
    orders = np.argsort(features[0, -1, [2, 4, 5]]).tolist()
    self.assertListEqual(orders, list(range(3)))
    orders = np.argsort(features[0, -1, [0, 1, 3]]).tolist()
    self.assertListEqual(orders, list(range(3)))

  @skip
  def test_fixed_kbody_terms(self):
    species = get_species({"C": 7, "H": 9, "N": 3})
    kbody_terms = sorted(list(set(
      [",".join(sorted(c)) for c in combinations(species, 3)])))
    num_terms = len(kbody_terms)

    coords = get_example(5)
    clf = kbody_transform.Transformer(
      get_species({"C": 1, "H": 4}),
      many_body_k=3,
      kbody_terms=kbody_terms
    )

    self.assertEqual(len(clf.split_dims), num_terms)
    self.assertEqual(len(clf.kbody_terms), num_terms)
    self.assertEqual(clf.cnk, int(comb(5, 3, exact=True)))

    features, _ = clf.transform(coords)
    self.assertTupleEqual(features.shape, (1, 18, 3))
    self.assertListEqual(clf.split_dims, [1, 1, 1, 6, 1, 1, 4, 1, 1, 1])
    self.assertAlmostEqual(np.sum(features[0, 0:3, :]), 0.0, delta=epsilon)

    d12 = np.linalg.norm(coords[0, 1, :] - coords[0, 2, :])
    s12 = 0.64 * 1.5
    self.assertAlmostEqual(features[0, 3, 2], np.exp(-d12 / s12), delta=epsilon)

    selection = clf.kbody_selections['C,H,H']
    self.assertEqual(len(selection), comb(4, 2, exact=True))
    self.assertListEqual(selection[0], [0, 1, 2])
    self.assertListEqual(selection[1], [0, 1, 3])
    self.assertListEqual(selection[2], [0, 1, 4])
    self.assertListEqual(selection[3], [0, 2, 3])
    self.assertListEqual(selection[4], [0, 2, 4])
    self.assertListEqual(selection[5], [0, 3, 4])

  def test_fixed_split_dims(self):
    many_body_k = 3
    occurs = {"C": 7, "H": 9, "O": 4}
    species = get_species(occurs)
    kbody_terms = sorted(list(set(
      [",".join(sorted(c)) for c in combinations(species, many_body_k)])))
    num_terms = len(kbody_terms)
    split_dims = []
    for term in kbody_terms:
      counter = Counter(term.split(","))
      dims = [comb(occurs[e], k, True) for e, k in counter.items()]
      split_dims.append(np.prod(dims))
    split_dims = [int(x) for x in split_dims]

    clf = kbody_transform.Transformer(
      get_species({"C": 2, "H": 4}),
      many_body_k=3,
      kbody_terms=kbody_terms,
      split_dims=split_dims
    )

    coords = get_example(6)
    features, _ = clf.transform(coords)
    offsets = [0] + np.cumsum(clf.split_dims).tolist()
    selections = clf.kbody_selections
    self.assertEqual(features.shape[1], comb(20, many_body_k, exact=True))

    ccc = features[0, offsets[0]: offsets[1], :]
    self.assertAlmostEqual(np.sum(ccc), 0.0, delta=epsilon)

    cch = features[0, offsets[1]: offsets[2], :]
    dists = pairwise_distances(coords[0, selections['C,C,H'][0]])
    dists = dists[[0, 0, 1], [1, 2, 2]]
    lmat = [1.5, 1.07, 1.07]
    vsum = np.exp(-dists / np.asarray(lmat) / 1.5).sum()
    self.assertAlmostEqual(vsum, cch[0].sum(), delta=epsilon)

    cco = features[0, offsets[2]: offsets[3], :]
    self.assertAlmostEqual(np.sum(cco), 0.0, delta=epsilon)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
class MultiTransformerTest(tf.test.TestCase):
  pass


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
class FixedLenMultiTransformerTest(tf.test.TestCase):

  def test_main(self):
    max_occurs = {"C": 3, "H": 4, "O": 3}
    max_natoms = sum(max_occurs.values())
    many_body_k = 3
    clf = kbody_transform.FixedLenMultiTransformer(
      max_occurs,
      many_body_k=many_body_k
    )

    species = get_species(max_occurs)
    coords = get_example(len(species))
    features, split_dims, _ = clf.transform(species, coords)
    total_dim = comb(max_natoms, many_body_k, True)

    self.assertTupleEqual(features.shape, (1, total_dim, 3))
    self.assertEqual(len(split_dims), 10)

    species = get_species({"C": 1, "H": 4})
    coords = get_example(len(species))
    features, split_dims_, _ = clf.transform(species, coords)
    self.assertListEqual(split_dims, split_dims_)
    self.assertTupleEqual(features.shape, (1, total_dim, 3))


if __name__ == "__main__":
  tf.test.main()
