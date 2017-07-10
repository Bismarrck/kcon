#!coding=utf-8
"""
The unit tests of `CNNPredictor`.
"""
import numpy as np
import unittest
from os.path import join, basename, splitext
from xyz import extract_xyz
from kbody_predict import CNNPredictor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


def measure_performance(graph_model_path, xyzfile, xyz_format, num_atoms,
                        num_examples, mixed=True):
  """
  Measure the performance of an exported model.

  Args:
    graph_model_path: a `str` as the path of the model.
    xyzfile: a `str` as the xyz file to parse.
    xyz_format: a `str` as the format of the xyz file.
    num_atoms: a `int` as the maximum number of atoms.
    num_examples: a `int` as the number of examples to parse.
    mixed: a `bool` indicating whether the model is trained for a mixed dataset
      or not.

  """
  clf = CNNPredictor(graph_model_path)
  xyz = extract_xyz(xyzfile, num_atoms=num_atoms, num_examples=num_examples,
                    xyz_format=xyz_format, verbose=False)
  xyz.split()

  samples = xyz.get_testing_samples()
  y_true = samples[1]

  if not mixed:
    species = samples[0][0]
    inputs = dict(zip(["array_of_coords", "array_of_lattice", "array_of_pbc"],
                      [samples[i] for i in (2, 4, 5)]))
    y_nn = clf.predict_total(species, **inputs)

  else:
    y_nn = np.zeros_like(y_true)
    for i in range(len(y_true)):
      species = samples[0][i]
      natom = len(species)
      coords = samples[2][i][:natom]
      y_nn[i] = clf.predict_total(
        species,
        coords,
        array_of_lattice=samples[4][i],
        array_of_pbc=samples[5][i]
      )

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


def test_tio2_dftb():
  """
  Measure the performance of the trained model of `TiO2.DFTB`.
  """
  num_atoms = 27
  num_examples = 5000
  graph_model_path = join("models", "TiO2.DFTB.v5.pb")
  xyzfile = join("..", "datasets", "TiO2.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_atoms, num_examples, mixed=False)


def test_quinoline_dft():
  """
  Measure the performance of the trained model of `C9H7N.PBE`.
  """
  num_atoms = 17
  num_examples = 5000
  graph_model_path = join("models", "C9H7N.PBE.v5.pb")
  xyzfile = join("..", "datasets", "C9H7N.PBE.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_atoms, num_examples, mixed=False)


def test_quinoline_dftb():
  """
  Measure the performance of the trained model of `C9H7Nv1`.
  """
  num_atoms = 17
  num_examples = 5000
  graph_model_path = join("models", "C9H7Nv1.DFTB.v5.pb")
  xyzfile = join("..", "datasets", "C9H7Nv1.xyz")
  xyz_format = "grendel"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_atoms, num_examples, mixed=False)


def test_qm7_dft():
  """
  Measure the performance of the trained model of `qm7`.
  """
  num_atoms = 23
  num_examples = 7165
  graph_model_path = join("models", "qm7.v5.pb")
  xyzfile = join("..", "datasets", "qm7.xyz")
  xyz_format = "xyz"
  measure_performance(
    graph_model_path, xyzfile, xyz_format, num_atoms, num_examples, mixed=True)


if __name__ == '__main__':
  unittest.main()
