from __future__ import print_function

from scipy.io import loadmat
import numpy as np

kcal_to_hartree = 1.0 / 627.509474
au_to_angstrom = 0.52917721092

z_to_symbols = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}

mat = loadmat("qm7.mat")
Z = mat['Z'].astype(np.int64).tolist()
R = mat['R'] * au_to_angstrom
T = mat['T'].flatten()
max_natoms = R.shape[1]

collections = {}

for i in range(len(Z)):
  z = Z[i]
  natoms = max_natoms - z.count(0)
  species = [z_to_symbols[zi] for zi in z[:natoms]]
  formula = ",".join(species)
  collections[formula] = collections.get(formula, []) + [(R[i, :natoms], T[i])]

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
      outs.append("{:2s} {: 14.8f} {: 14.8f} {: 14.8f}".format(elements[i], *xyz))

with open("qm7.xyz", "w") as f:
  f.write("\n".join(outs))

