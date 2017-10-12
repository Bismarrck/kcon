# coding=utf-8
"""
A complete test suit for transforming force-related features.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import ase.io
from transformer import Transformer
from constants import GHOST, pyykko
from itertools import combinations
from os.path import join, dirname
from tensorflow.contrib.layers.python.layers import flatten

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def print_atoms(atoms):
  """
  Print the coordinates of the `ase.Atoms` in XYZ format.
  """
  for i, atom in enumerate(atoms):
    print("{:2d} {:2s} {: 14.8f} {: 14.8f} {: 14.8f}".format(
      i, atom.symbol, *atom.position))


def array2string(x):
  """
  Convert the numpy array to a string.
  """
  return np.array2string(x, formatter={"float": lambda f: "%-8.3f" % f})


def get_test_examples():
  """
  Return the example molecules to test.

  Returns:
    list_of_atoms: a `list` of `ase.Atoms` objects.

  """
  filename = join(dirname(__file__), "..", "test_files", "transform_forces.xyz")
  return ase.io.read(filename, index=":", format="xyz")


def get_conditional_sorting_indices(bonds):
  """
  Return the columns that should be sorted.

  Args:
    bonds: a list of `str` as the ordered bond types for a k-body term.

  Returns:
    columns: a list of `int` as the indices of the columns to sort.

  """
  columns = []
  if bonds[0] == bonds[1]:
    if bonds[1] == bonds[2]:
      columns = [0, 1, 2]
    else:
      columns = [0, 1]
  elif bonds[0] == bonds[2]:
    columns = [0, 2]
  elif bonds[1] == bonds[2]:
    columns = [1, 2]
  return columns


def get_coef_naive(transformer, atoms, max_occurs=None):
  """
  The straightforward and naive way to compute the coefficients matrix.

  Args:
    transformer: a `Transformer`.
    atoms: an `ase.Atoms` object.
    max_occurs: a `dict` as the maximum appearance for each element.

  Returns:
    coefficients: the coefficients matrix.

  """

  symbols = atoms.get_chemical_symbols()
  positions = atoms.positions
  bond_types = transformer.get_bond_types()

  # `n_max` is the number of atoms including the ghost atoms while `n_real` is
  # the number of real atoms in the given `ase.Atoms` object.
  if max_occurs is None:
    n_real = len(atoms)
  else:
    n_max = sum(max_occurs.values())
    n_real = n_max - max_occurs.get(GHOST, 0)

  # The shape of the `coefficients` is determined by `n_real` when `k_max` is 3.
  # We may have some dummy or virtual entries.
  coefficients = np.zeros((n_real * 3, (n_real - 1) ** 2))
  locations = np.zeros(n_real, dtype=int)

  vlist = np.zeros(3)
  xlist = np.zeros(3)
  ylist = np.zeros(3)
  zlist = np.zeros(3)
  ilist = np.zeros(3, dtype=int)
  jlist = np.zeros(3, dtype=int)

  for kbody_term in transformer.kbody_terms:
    if kbody_term not in transformer.kbody_selections:
      continue

    selections = transformer.kbody_selections[kbody_term]
    columns = get_conditional_sorting_indices(bond_types[kbody_term])

    for selection in selections:

      vlist.fill(0.0)
      xlist.fill(0.0)
      ylist.fill(0.0)
      zlist.fill(0.0)

      # (i, j) is the indices of the atoms that form the bond r_ij.
      for k, (i, j) in enumerate(combinations(selection, r=2)):
        if i >= len(atoms) or j >= len(atoms):
          continue
        r = atoms.get_distance(i, j)
        radius = pyykko[symbols[i]] + pyykko[symbols[j]]
        v = np.exp(-r / radius)
        g = (1.0 / radius**2) * v * (positions[i] - positions[j]) / np.log(v)
        assert isinstance(g, np.ndarray)

        vlist[k] = v
        ilist[k] = i
        jlist[k] = j
        xlist[k] = g[0]
        ylist[k] = g[1]
        zlist[k] = g[2]

      # Apply the conditional sorting algorithm if `columns` is not empty.
      if len(columns) > 0:
        orders = np.argsort(vlist[columns])
        for vector in (ilist, jlist, xlist, ylist, zlist):
          entries = vector[columns]
          entries = entries[orders]
          vector[columns] = entries

      # Assign the calculated coefficients.
      for k in range(3):
        i = ilist[k]
        j = jlist[k]
        x = xlist[k]
        y = ylist[k]
        z = zlist[k]
        coefficients[i * 3: i * 3 + 3, locations[i]] = +x, +y, +z
        coefficients[j * 3: j * 3 + 3, locations[j]] = -x, -y, -z
        locations[i] += 1
        locations[j] += 1

  return coefficients


def reorder(coefficients, indexing):
  """
  Re-order the given `coefficients` matrix using the `indexing` matrix.
  """
  c = np.pad(coefficients.flatten(), [[1, 0]], mode='constant')
  return c[indexing]


def batch_reorder(g, indexing):
  """
  The batch reordering algorithm implemented in `inference_forces`.

  Args:
    g: a 3D Tensor as the elementwise multiplication results of the auxiliary
      coefficients and `dydz` for computing atomic forces.
    indexing: a 3D Tensor as the indexing matrix for force compoenents.

  Returns:
    forces: a `float32` Tensor of shape `[-1, num_force_components]` as the
      computed atomic forces.

  """
  g = tf.constant(g, dtype=tf.float32, name="tiled")
  indexing = tf.constant(indexing, dtype=tf.int32, name="indexing")

  with tf.name_scope("Forces"):

    # Now we should re-order all entries of `g`. Flatten it so that its shape
    # will be `[-1, D * 6 * C(k, 2)]`.
    g = flatten(g)
    g = tf.pad(g, [[0, 0], [1, 0]], mode='constant', name="pad")

    # The basic idea of the re-ordering algorithm is taking advantage of the
    # array broadcasting scheme of TensorFlow (Numpy). Since the batch size (the
    # first axis of `g`) will not be 1, we cannot do broadcasting directly.
    # Instead, we make the `g` a total flatten vector and broadcast it into a
    # matrix with `indexing`.
    with tf.name_scope("reshape"):

      with tf.name_scope("indices"):

        with tf.name_scope("g"):
          shape = tf.shape(g, name="shape")
          batch_size, step = shape[0], shape[1]

        with tf.name_scope("indexing"):
          shape = tf.shape(indexing, name="shape")
          num_f, num_entries = shape[1], shape[2]

        multiples = [1, num_f, num_entries]
        size = tf.multiply(batch_size, step, name="total_size")
        steps = tf.range(0, limit=size, delta=step, name="arange")
        steps = tf.reshape(steps, (batch_size, 1, 1), name="steps")
        steps = tf.tile(steps, multiples, name="tiled")
        indexing = tf.add(indexing, steps, name="indices")

      # Do the broadcast
      g = tf.reshape(g, (-1, ), "1D")

      # Pad an zero at the begining of the totally flatten `g` because real
      # indices in `indexing` start from one and the index of zero suggests the
      # contribution should also be zero.
      # g = tf.pad(g, [[1, 0]], name="pad")
      g = tf.gather(g, indexing, name="gather")

      # Reshape `g` so that all entries of each row (axis=2) correspond to the
      # same force component (axis=1).
      g = tf.reshape(g, (batch_size, num_f, num_entries), "reshape")

    # Sum up all entries of each row to get the final gradient for each force
    # component.
    g = tf.reduce_sum(g, axis=2, keep_dims=False, name="sum")

    # Always remember the physics law: f = -dE / dr. But the output `y_total`
    # already took the minus sign.
    g = tf.identity(g, "forces")

  with tf.Session() as sess:
    return sess.run(g)


def eval_row_diff(source, target):
  """
  Return the sum of absolute differences of each row.
  """
  diff = np.zeros(len(source))
  for i in range(len(source)):
    diff[i] = np.abs(np.sort(source[i]) - np.sort(target[i])).sum()
  return diff


def eval_all_diff(source, target):
  """
  Return the sum of absolute differences of all entries.
  """
  return np.abs(np.sort(source.flatten()) - np.sort(target.flatten())).sum()


class TransformerTests(tf.test.TestCase):

  def setUp(self):
    """
    Load the `ase.Atoms` objects.
    """
    list_of_atoms = get_test_examples()
    self.atoms = {
      "C10H8": [list_of_atoms[0]],
      "CH4": [list_of_atoms[1]],
      "C2H6": [list_of_atoms[2]],
      "C6H3N": [list_of_atoms[3]],
      "C2H6O": list_of_atoms[4:9]
    }
    self.epsilon = 1e-3

  def _eval_simple(self, name, list_of_atoms):
    """
    Evaluate the computed coefficients using the default `Transformer`.
    """
    print("Test {} using default `Transformer` ...".format(name))
    clf = Transformer(list_of_atoms[0].get_chemical_symbols(),
                      atomic_forces=True)
    for atoms in list_of_atoms:
      _, transformed, indexing = clf.transform(atoms)
      target = reorder(transformed, indexing)
      source = get_coef_naive(clf, atoms)
      row_diff = eval_row_diff(source, target)
      all_diff = eval_all_diff(source, target)
      self.assertLess(all_diff, self.epsilon)
      self.assertLess(max(row_diff), self.epsilon**2)

  def test_simple_methane(self):
    self._eval_simple("Methane", self.atoms["CH4"])

  def test_simple_ethane(self):
    self._eval_simple("Ethane", self.atoms["CH4"])

  def test_simple_c6h3n(self):
    self._eval_simple("C6H3N", self.atoms["C6H3N"])

  def test_simple_napthalene(self):
    self._eval_simple("Napthalene", self.atoms["C10H8"])

  def test_simple_ethanol(self):
    self._eval_simple("Ethanol", self.atoms["C2H6O"])

  def test_simple_batch_reorder(self):
    list_of_atoms = self.atoms["C2H6O"]
    n = len(list_of_atoms)

    clf = Transformer(list_of_atoms[0].get_chemical_symbols(),
                      atomic_forces=True)
    n_rows, n_cols = clf.shape
    n_f_components = clf.num_force_components
    n_entries = clf.num_entries_per_component
    array_of_coefficients = np.zeros((n, n_rows, n_cols * 6))
    array_of_indexing = np.zeros((n, n_f_components, n_entries), dtype=int)
    singles = []

    for i, atoms in enumerate(list_of_atoms):
      _, coefficients, indexing = clf.transform(atoms)
      array_of_coefficients[i] = coefficients
      array_of_indexing[i] = indexing
      singles.append(reorder(coefficients, indexing).sum(axis=1))

    batches = batch_reorder(array_of_coefficients, array_of_indexing)

    for i in range(len(batches)):
      diff = np.abs(singles[i] - batches[i])
      self.assertLess(diff.max(), self.epsilon)


if __name__ == "__main__":
  tf.test.main()
