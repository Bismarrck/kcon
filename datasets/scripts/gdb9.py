from __future__ import print_function

import numpy as np
import re
import sys
from scipy.misc import comb
from os.path import join
from os import listdir


def get_regex_pattern():
  return re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")


def extract_xyz(filename, verbose=False):
  energy = 0
  coords = []
  species = []
  stage = 0
  num_atoms = None
  xyz_patt = get_regex_pattern()
  j = 0
  with open(filename) as f:
    for line in f:
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          num_atoms = int(l)
          if num_atoms > 1:
            stage += 1
      elif stage == 1:
        elements = l.split("\t")
        if len(elements) == 16:
          # Internal energy at 298.15K
          energy = float(elements[12])
          stage += 1
      elif stage == 2:
        l = l.replace("\t", " ")
        m = xyz_patt.search(l)
        if m:
          coords.append([float(v) for v in m.groups()[1:4]])
          species.append(m.group(1))
          j += 1
          if j == num_atoms:
            break
  return species, energy, coords


def to_xyz_strings(species, energy, coords):
  fmt = "{:2s}  {: 14.8f} {: 14.8f} {: 14.8f}"
  n = len(species)
  strings = ["{:d}".format(n), "{:.8f}".format(energy)]
  strings.extend([fmt.format(species[i], *coords[i]) for i in range(n)])
  return strings


with open("gdb9.xyz", "w+") as f:

  array_of_species = []
  coordinates = []
  energies = []
  workdir = "./dsgdb9nsd"
  ntotal = 133885
  coef = np.zeros((ntotal, 7))
  ordered = ["C", "H", "N", "O", "F"]

  for i in range(ntotal):
    xyzfile = "dsgdb9nsd_{:06d}.xyz".format(i + 1)
    species, energy, coords = extract_xyz(join(workdir, xyzfile), verbose=False)
    array_of_species.append(species)
    energies.append(energy)
    coordinates.append(coords)
    natom = len(species)
    for j, atom in enumerate(ordered):
      coef[i, j] = species.count(atom)
    coef[i, 5] = comb(natom, 2, exact=True)
    coef[i, 6] = comb(natom, 3, exact=True)

    if i > 0 and i % 1000 == 0:
      sys.stdout.write("\rProgress: %7d / %7d" % (i, 133885))
  sys.stdout.write("\n")

  b = np.array(energies)
  x = np.linalg.pinv(coef) @ b
  residual = np.abs(coef @ x - b)

  print("Linear Fit Residual Avg: {:.6f}".format(residual.mean()))
  print("Linear Fit Residual Min: {:.6f}".format(residual.min()))
  print("Linear Fit Residual Max: {:.6f}".format(residual.max()))

  outliers = list(np.where(residual > 10 * residual.mean())[0])
  print("Number of outliers: {:d}".format(len(outliers)))

  outputs = []
  ntotal = len(array_of_species)
  for i in range(ntotal):
    if i in outliers:
      continue
    outputs.extend(to_xyz_strings(array_of_species[i], energies[i], coordinates[i]))
  f.write("\n".join(outputs))
