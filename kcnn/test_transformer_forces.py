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


def eval_diff(source, target):
  """
  Return the sum of absolute differences of each row.
  """
  diff = np.zeros(len(source))
  for i in range(len(source)):
    diff[i] = np.abs(np.sort(source[i]) - np.sort(target[i])).sum()
  return diff


class TransformerTests(tf.test.TestCase):

  def setUp(self):
    """
    Load the `ase.Atoms` objects.
    """
    list_of_atoms = get_test_examples()
    self.atoms = {
      "C10H8": list_of_atoms[0],
      "CH4": list_of_atoms[1],
      "C2H6": list_of_atoms[2],
      "C6H3N": list_of_atoms[3],
      "C2H6O": list_of_atoms[4]
    }
    self.threshold = 1e-6

  def _eval_simple(self, name, atoms):
    """
    Evaluate the computed coefficients using the default `Transformer`.
    """
    print("Test {} using default `Transformer` ...".format(name))
    clf = Transformer(atoms.get_chemical_symbols(), atomic_forces=True)
    _, transformed, indexing = clf.transform(atoms)
    target = reorder(transformed, indexing)
    source = get_coef_naive(clf, atoms)
    diff = eval_diff(source, target)
    self.assertLess(max(diff), self.threshold)

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


if __name__ == "__main__":
  tf.test.main()
