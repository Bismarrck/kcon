# coding=utf-8
"""
This script is used to build the QM7 dataset from the origin .mat file.
"""
from __future__ import print_function

from scipy.io import loadmat
import numpy as np

kcal_to_hartree = 1.0 / 627.509474
au_to_angstrom = 0.52917721092
z_to_symbols = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def load(matfile):
  """
  Load the raw MATLAB data file.

  Args:
    matfile: a `str` as the matlab data file to read.

  """
  mat = loadmat(matfile)
  Z = mat['Z'].astype(np.int64).tolist()
  R = mat['R'] * au_to_angstrom
  T = mat['T'].flatten()
  max_natoms = R.shape[1]
  return Z, R, T, max_natoms


def build_qm7(matfile):
  """
  The main function to build the QM7 dataset.

  Args:
    matfile: a `str` as the matlab data file to read.

  """
  Z, R, T, max_natoms = load(matfile)

  collections = {}
  for i in range(len(Z)):
    z = Z[i]
    natoms = max_natoms - z.count(0)
    species = [z_to_symbols[zi] for zi in z[:natoms]]
    formula = ",".join(species)
    collections[formula] = collections.get(formula, []) + [
      (R[i, :natoms], T[i])
    ]

  outs = []
  for formula in collections:
    elements = formula.split(",")
    natoms = len(elements)
    for coords, energy in collections[formula]:
      outs.extend([
        "{:d}".format(natoms),
        "{:.8f}".format(energy * kcal_to_hartree)
      ])
      for i, xyz in enumerate(coords):
        outs.append("{:2s} {: 14.8f} {: 14.8f} {: 14.8f}".format(
          elements[i], *xyz))
  with open("qm7.xyz", "w") as f:
    f.write("\n".join(outs))


if __name__ == "__main__":
  build_qm7(matfile)


