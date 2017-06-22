#!coding=utf-8
"""
This module defines the utility function to extract xyz files.
"""
from __future__ import print_function, absolute_import

import re
import sys
import time
import numpy as np
from constants import hartree_to_ev, SEED
from collections import namedtuple, Counter
from sklearn.model_selection import train_test_split


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


XyzFormat = namedtuple("XyzFormat", (
  "name", "energy_patt", "string_patt", "default_unit", "parse_forces")
)

_grendel = XyzFormat(
  name="grendel",
  energy_patt=re.compile(r"Lattice=\"(.*)\".*"
                         r"energy=([\d.-]+)\s+pbc=\"(.*)\""),
  string_patt=re.compile(r"([A-Za-z]{1,2})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
                         r"\s+\d+\s+\d.\d+\s+\d+\s+([\d.-]+)\s+([\d.-]+)\s+"
                         r"([\d.-]+)"),
  default_unit=1.0,
  parse_forces=True
)

_xyz = XyzFormat(
  name="xyz",
  energy_patt=re.compile(r"([\w.-]+)"),
  string_patt=re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)"),
  default_unit=hartree_to_ev,
  parse_forces=False,
)

_cp2k = XyzFormat(
  name="cp2k",
  energy_patt=re.compile(r"i\s=\s+\d+,\sE\s=\s+([\w.-]+)"),
  string_patt=re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)"),
  default_unit=hartree_to_ev,
  parse_forces=False,
)

_extxyz = XyzFormat(
  name="extxyz",
  energy_patt=re.compile(r"i=(\d+).(\d+),\sE=([\d.-]+)"),
  string_patt=re.compile(r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)"),
  default_unit=hartree_to_ev,
  parse_forces=False,
)


def _get_regex_patt_and_unit(xyz_format):
  """
  Return the corresponding regex patterns and the energy unit.

  Args:
    xyz_format: a `str` as the format of the file.

  Returns:
    formatter: a `XyzFormat`.

  """
  if xyz_format.lower() == 'grendel':
    formatter = _grendel
  elif xyz_format.lower() == 'cp2k':
    formatter = _cp2k
  elif xyz_format.lower() == 'xyz':
    formatter = _xyz
  elif xyz_format.lower() == 'extxyz':
    formatter = _extxyz
  else:
    raise ValueError("The file format of %s is not supported!" % xyz_format)
  return formatter


class XyzFile:
  """
  This class is used to managed the parsed data from a xyz file.
  """

  def __init__(self, array_of_species, energies, array_of_coords,
               array_of_forces, array_of_lattice, array_of_pbc):
    """
    Initialization method.

    Args:
      array_of_species: a `List[List[str]]` as the species of structures.
      energies: a `float64` array of shape `[-1]` as the total energies of the
        structures.
      array_of_coords: a `float32` array of shape `[-1, N, 3]` as the atomic
        coordinates of the structures.
      array_of_forces: a `float32` array of shape `[-1, N, 3]` as the atomic
        forces of the structures.
      array_of_lattice: a `float32` array of shape `[-1, 9]` as the lattice
        parameters for structures.
      array_of_pbc: a `bool` array of shape `[-1, 3]` as the periodic conditions
        along X,Y,Z directions for structures.

    """
    self._array_of_species = array_of_species
    self._energies = energies
    self._array_of_coords = array_of_coords
    self._array_of_forces = array_of_forces
    self._array_of_lattice = array_of_lattice
    self._array_of_pbc = array_of_pbc
    self._num_examples = len(array_of_species)
    self._num_atoms = max(map(len, array_of_species))
    self._splitted = False
    self._trains = None
    self._tests = None
    self._random_state = SEED
    self._data = (array_of_species, energies, array_of_coords, array_of_forces,
                  array_of_lattice, array_of_pbc)

  @property
  def num_examples(self):
    """
    Return the total number of examples in this dataset.
    """
    return self._num_examples

  @property
  def num_atoms(self):
    """
    Return the maximum number of atoms of structures in this dataset.
    """
    return self._num_atoms

  @property
  def random_state(self):
    """
    Return the random state used to split this dataset.
    """
    return self._random_state

  @property
  def energy_range(self):
    """
    Return a tuple of two floats as the minimum and maximum energy in this
    dataset.
    """
    return min(self._energies), max(self._energies)

  @property
  def indices_of_training(self):
    """
    Return the indices of the training samples.
    """
    return self._trains

  @property
  def indices_of_testing(self):
    """
    Return the indices of the testing samples.
    """
    return self._tests

  def __getitem__(self, i):
    """
    x.__getitem__(i) <=> x[i]

    Args:
      i: a `int` or `List[int]` as the indices of the samples to get.

    Returns:
      y: a `tuple` of samples.

    """
    return tuple(x[i] for x in self._data)

  def get_max_occurs(self):
    """
    Return the maximum occurances for each type of atom.

    Returns:
      max_occurs: a `dict` as the maximum occurances for each type of atom.

    """
    max_occurs = {}
    for species in self._array_of_species:
      c = Counter(species)
      for specie, times in c.items():
        max_occurs[specie] = max(max_occurs.get(specie, 0), times)
    return max_occurs

  def split(self, test_size=0.2, random_state=None):
    """
    Split this dataset into training set and testing set.

    Args:
      test_size: a `float` or `int`. If float, should be between 0.0 and 1.0 and
        represent the proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples.
      random_state: a `int` as the pseudo-random number generator state used for
        random sampling.

    """
    random_state = random_state or self._random_state
    indices = range(self._num_examples)
    trains, tests = train_test_split(
      indices, test_size=test_size, random_state=random_state)
    self._random_state = random_state
    self._splitted = True
    self._trains = trains
    self._tests = tests

  def get_testing_samples(self):
    """
    Return the testing samples.
    """
    if not self._splitted:
      self.split()
    return self[self._tests]

  def get_training_samples(self):
    """
    Return the training samples.
    """
    if not self._splitted:
      self.split()
    return self[self._trains]


def extract_xyz(filename, num_examples, num_atoms, xyz_format='xyz',
                verbose=True, energy_to_ev=None):
  """
  Extract atomic species, energies, coordiantes, and perhaps forces, from the
  file.

  Args:
    filename: a `str` as the file to parse.
    num_examples: a `int` as the maximum number of examples to parse.
    num_atoms: a `int` as the number of atoms. If `mixed` is True, this should
      be the maximum number of atoms in one configuration.
    xyz_format: a `str` representing the format of the given xyz file.
    verbose: a `bool` indicating whether we should log the parsing progress.
    energy_to_ev: a `float` as the unit for converting energies to eV. Defaults
      to None so that default units will be used.

  Returns
    xyz: a `XyzFile` as the parsed results.

  """

  energies = np.zeros((num_examples,), dtype=np.float64)
  array_of_coords = np.zeros((num_examples, num_atoms, 3), dtype=np.float32)
  array_of_forces = np.zeros((num_examples, num_atoms, 3), dtype=np.float32)
  array_of_lattice = np.zeros((num_examples, 9))
  array_of_pbc = np.zeros((num_examples, 3), dtype=bool)
  array_of_species = []
  species = []
  stage = 0
  count = 0
  ai = 0
  natoms = None
  formatter = _get_regex_patt_and_unit(xyz_format)
  assert isinstance(formatter, XyzFormat)
  unit = energy_to_ev or formatter.default_unit

  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")
  with open(filename) as f:
    for line in f:
      if count == num_examples:
        break
      l = line.strip()
      if l == "":
        continue
      # The first stage: parsing the number of atoms of next structure. The
      # parsed `natoms` should be not larger than `num_atoms` because we already
      # allocated memeory spaces.
      if stage == 0:
        if l.isdigit():
          natoms = int(l)
          if natoms > num_atoms:
            raise ValueError("The number of atoms %d from the file is larger "
                             "than the given maximum %d!" % (natoms, num_atoms))
          stage += 1
      # The second stage is to parse the total energy and other properties which
      # depends on the file format. All energies are converted to eV.
      elif stage == 1:
        m = formatter.energy_patt.search(l)
        if m:
          if xyz_format.lower() == 'extxyz':
            energies[count] = float(m.group(3)) * unit
          elif xyz_format.lower() == 'grendel':
            energies[count] = float(m.group(2)) * unit
            array_of_lattice[count] = [float(x) for x in m.group(1).split()]
            array_of_pbc[count] = [True if x == "T" else False
                                   for x in m.group(3).split()]
          else:
            energies[count] = float(m.group(1)) * unit
          stage += 1
      # The third stage is to parse atomic symbols and coordinates. If the file
      # format is `grendel` the forces are also parsed.
      elif stage == 2:
        m = formatter.string_patt.search(l)
        if m:
          array_of_coords[count, ai, :] = [float(v) for v in m.groups()[1:4]]
          if formatter.parse_forces:
            array_of_forces[count, ai, :] = [float(v) for v in m.groups()[4:7]]
          species.append(m.group(1))
          ai += 1
          if ai == natoms:
            array_of_species.append(species)
            species = []
            ai = 0
            stage = 0
            count += 1
            if verbose and count % 1000 == 0:
              sys.stdout.write(
                "\rProgress: %7d  /  %7d" % (count, num_examples))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  # Resize all arrays if `count` is smaller than `num_examples`.
  array_of_species = np.asarray(array_of_species)
  if count < num_examples:
    energies = np.resize(energies, (count, ))
    array_of_coords = np.resize(array_of_coords, (count, num_atoms, 3))
    array_of_forces = np.resize(array_of_forces, (count, num_atoms, 3))
    array_of_lattice = np.resize(array_of_lattice, (count, 9))
    array_of_pbc = np.resize(array_of_pbc, (count, 3))

  return XyzFile(array_of_species, energies, array_of_coords, array_of_forces,
                 array_of_lattice, array_of_pbc)
