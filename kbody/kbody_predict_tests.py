#!coding=utf-8
"""
The unit tests of `CNNPredictor`.
"""
import numpy as np
import unittest
from os.path import join, dirname
from kbody_input import extract_xyz
from kbody_predict import CNNPredictor


def _print_predictions(y_total, y_true, y_atomic, species):
  """
  A helper function for printing predicted results of the unittests.

  Args:
    y_total: a 1D array of shape [N, ] as the predicted energies.
    y_true: a 1D array of shape [N, ] as the real energies.
    y_atomic: a 2D array of shape [N, M] as the atomic energies.
    species: a `List[str]` as the atomic species.

  """
  num_examples, num_atoms = y_atomic.shape
  size = min(num_examples, 20)
  y_total = np.atleast_1d(y_total)
  y_true = np.atleast_1d(y_true)
  for i in np.random.choice(range(num_examples), size=size):
    print("Index            : % 2d" % i)
    print("Energy Predicted : % .4f eV" % y_total[i])
    print("Energy Real      : % .4f eV" % y_true[i])
    for j in range(num_atoms):
      print("Atom %2d, %2s,     % 10.4f eV" % (j, species[j], y_atomic[i, j]))
    print("")


def test_quinoline_dft():
  """
  Test the trained model of C9H7N.
  """
  graph_model_path = join(
    dirname(__file__), "models", "C9H7N.PBE.v5", "C9H7N.PBE-1000000.pb")
  calculator = CNNPredictor(graph_model_path)

  print("------------")
  print("Tests: C9H7N")
  print("------------")

  xyzfile = join(dirname(__file__), "..", "datasets", "C9H7N.PBE.xyz")
  samples = extract_xyz(
    xyzfile, num_examples=5000, num_atoms=17, xyz_format='grendel')

  species = samples[0][0]
  y_true = float(samples[1][0])
  coords = samples[2][0]

  y_total, _, y_atomic, _ = calculator.predict(species, coords)
  _print_predictions(y_total, [y_true], y_atomic, species)


if __name__ == '__main__':
  unittest.main()
