# coding=utf-8
"""
This script is used to process xyz files of the MD datasets downloaded from:
http://quantum-machine.org/datasets/#md-datasets
"""
from __future__ import print_function

import numpy as np
from os import listdir
from os.path import isdir, isfile, join, basename


lattcie = "\"20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0\""
properties = "species:S:1:pos:R:3:Z:I:1:magmoms:R:1:tags:I:1:forces:R:3"
pbc = "\"F F F\""
kcal_to_ev = 1.0 / 23.061
symbol_to_number = {
  "C": 6,
  "H": 1,
  "N": 7,
  "O": 8
}


def parse_and_convert_file(filename):
  """
  Parse the given xyz file.

  Args:
    filename: a `str` as the file to parse.

  Returns:
    xyz: an `ase` type xyz string for the input structure.

  """
  stage = 0
  n = None
  energy = None
  forces = None
  symbols = []
  coords = []
  counter = 0
  
  with open(filename) as f:
    for line in f:
      l = line.strip()
      if stage == 0:
        if l.isdigit():
          n = int(l)
          stage += 1
      elif stage == 1:
        elements = l.split(";")
        if len(elements) == 2:
          energy = float(elements[0]) * kcal_to_ev
          forces = [[float(x) for x in g.split(",")] 
                    for g in elements[1][1:-1].split("],[")]
          if len(forces) != n:
            raise IOError()
          forces = np.array(forces) * kcal_to_ev
          stage += 1
      elif stage == 2:
        elements = l.split("\t")
        if len(elements) == 4:
          symbols.append(elements[0].strip())
          coords.append([float(c) for c in elements[1:]])
          counter += 1
        if counter == n:
          coords = np.array(coords)
          break

  outputs = ["{:d}".format(n)]
  outputs.append("Lattice={} Properties={} energy={} pbc={}".format(
    lattcie, properties, energy, pbc))
  for i in range(n):
    outputs.append(
      "{:2s} {: 14.8f} {: 14.8f} {: 14.8f} {:3d}"
      "    0.00000000        0 {: 14.8f} {: 14.8f} {: 14.8f}".format(
        symbols[i],
        coords[i, 0], coords[i, 1], coords[i, 2],
        symbol_to_number[symbols[i]],
        forces[i, 0], forces[i, 1], forces[i, 2]
      )
    )
  return "\n".join(outputs)


def build(directory, max_examples=None):
  """
  Build the dataset.

  Args:
    directory: a `str` as the unzipped dir.
    max_examples: an `int` as the maximum number of examples to use.

  """

  dataset = []
  max_examples = max_examples or np.inf
  count = 0

  if isdir(directory):
    for afile in listdir(directory):  
      if afile.startswith("."):
        continue
      string = parse_and_convert_file(join(directory, afile))
      dataset.append(string)
      count += 1
      if count == max_examples:
        break
  nk = int(count / 1000)
  with open(basename(directory) + "{}k.xyz".format(nk), "w+") as f:
    f.write("\n".join(dataset))


if __name__ == "__main__":
  build("./naphthalene", max_examples=10000)


