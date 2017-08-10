#!coding=utf-8
"""
The unit tests of `CNNPredictor`.
"""
import numpy as np
import unittest
from os.path import join, basename, splitext
from database import Database
from predictor import KcnnPredictor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


def measure_performance(graph_model_path, xyzfile, xyz_format, num_examples,
                        mixed=True):
  """
  Measure the performance of an exported model.

  Args:
    graph_model_path: a `str` as the path of the model.
    xyzfile: a `str` as the xyz file to parse.
    xyz_format: a `str` as the format of the xyz file.
    num_examples: a `int` as the number of examples to parse.
    mixed: a `bool` indicating whether the model is trained for a mixed dataset
      or not.

  """
  clf = KcnnPredictor(graph_model_path)

  database = Database.from_xyz(xyzfile,
                               num_examples=num_examples,
                               xyz_format=xyz_format)
  database.split()

  if not mixed:
    trajectory = [atoms for atoms in database.examples(for_training=False)]
    y_nn = clf.predict_total_energy(trajectory)
    y_true = [atoms.get_total_energy() for atoms in trajectory]

  else:
    ids = database.ids_of_testing_examples
    y_nn = np.zeros(len(ids))
    y_true = np.zeros_like(y_nn)
    for i in range(len(ids)):
      atoms = database[ids[i]]
      y_nn[i] = clf.predict_total_energy(atoms)
      y_true[i] = atoms.get_total_energy()

  y_true = np.asarray(y_true)
  y_diff = np.abs(y_true - y_nn)
  score = r2_score(y_true, y_nn)
  stddev = np.std(y_true - y_nn)
  mae = mean_absolute_error(y_true, y_nn)
  rmse = np.sqrt(mean_squared_error(y_true, y_nn))
  emin = y_diff.min()
  emax = y_diff.max()

  print("{}".format(splitext(basename(xyzfile))[0]))
  print("  * Model       : {}".format("v5"))
  print("  * R2 Score    : {: 8.6f}".format(score))
  print("  * MAE    (eV) : {: 8.3f}".format(mae))
  print("  * RMSE   (eV) : {: 8.3f}".format(rmse))
  print("  * Stddev (eV) : {: 8.3f}".format(stddev))
  print("  * Min    (eV) : {: 8.3f}".format(emin))
  print("  * Max    (eV) : {: 8.3f}".format(emax))
  print("End")


@unittest.skip
def test_tio2_dftb():
  """
  Measure the performance of the trained model of `TiO2.DFTB`.
  """
  num_examples = 5000
  graph_model_path = join("models", "TiO2.DFTB.v5.pb")
  xyzfile = join("..", "datasets", "TiO2.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_examples, mixed=False)


def test_quinoline_dft():
  """
  Measure the performance of the trained model of `C9H7N.PBE`.
  """
  num_examples = 5000
  graph_model_path = join("events", "freeze", "C9H7N.PBE-2002.pb")
  xyzfile = join("..", "datasets", "C9H7N.PBE.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_examples, mixed=False)


@unittest.skip
def test_quinoline_dftb():
  """
  Measure the performance of the trained model of `C9H7Nv1`.
  """
  num_examples = 5000
  graph_model_path = join("models", "C9H7Nv1.DFTB.v5.pb")
  xyzfile = join("..", "datasets", "C9H7Nv1.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_examples, mixed=False)


@unittest.skip
def test_qm7_dft():
  """
  Measure the performance of the trained model of `qm7`.
  """
  num_examples = 7165
  graph_model_path = join("models", "qm7.v5.pb")
  xyzfile = join("..", "datasets", "qm7.xyz")
  xyz_format = "xyz"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_examples, mixed=True)


if __name__ == '__main__':
  unittest.main()
