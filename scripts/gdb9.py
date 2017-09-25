# coding=utf-8
"""
This script is used to pre-process the xyz files of the dataset GDB-9.
"""
from __future__ import print_function

import numpy as np
import re
import sys
from scipy.misc import comb
from os.path import join
from os import listdir


def get_regex_pattern():
  """
  Return the regex pattern to extract atomic symbol and coordinates.
  """
  return re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")


def extract_xyz(filename, verbose=False):
  """
  Extract a raw xyz file of the GDB-9 dataset.

  Args:
    filename: a `str` as the file to parse.
    verbose: a `bool`.

  Returns:
    species: a list of `str` as the chemical symbols.
    energy: a `float` as the total energy of the structure.
    coords" a list of list of floats as the coordinates.

  """
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
  """
  Convert the structure to a standard xyz string.
  """
  fmt = "{:2s}  {: 14.8f} {: 14.8f} {: 14.8f}"
  n = len(species)
  strings = ["{:d}".format(n), "{:.8f}".format(energy)]
  strings.extend([fmt.format(species[i], *coords[i]) for i in range(n)])
  return strings


def build_gdb9(raw_dir):
  """
  The main function to build the GDB-9 dataset.

  Args:
    raw_dir: the directory of the extracted GDB-9 files.

  """
  with open("gdb9.xyz", "w+") as f:

    array_of_species = []
    coordinates = []
    energies = []
    ntotal = 133885
    coef = np.zeros((ntotal, 7))
    ordered = ["C", "H", "N", "O", "F"]

    for i in range(ntotal):
      xyzfile = "dsgdb9nsd_{:06d}.xyz".format(i + 1)
      species, energy, coords = extract_xyz(
        join(raw_dir, xyzfile), verbose=False
      )
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

    # Remove outliers. The origin GDB-9 dataset contains some outliers.
    # Some atoms are missing. So we need to remove these structures.
    b = np.array(energies)
    x = np.linalg.pinv(coef) @ b
    residual = np.abs(coef @ x - b)
    outliers = list(np.where(residual > 10 * residual.mean())[0])

    outputs = []
    ntotal = len(array_of_species)
    for i in range(ntotal):
      if i in outliers:
        continue
      out = to_xyz_strings(array_of_species[i], energies[i], coordinates[i])
      outputs.extend(out)
    f.write("\n".join(outputs))


if __name__ == "__main__":
  build_gdb9(raw_dir='./dsgdb9nsd')


