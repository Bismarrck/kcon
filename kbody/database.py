#!coding=utf-8
"""
This module defines the utility function to extract xyz files.
"""
from __future__ import print_function, absolute_import

import re
import sys
import time
import numpy as np
from ase.atoms import Atom, Atoms
from ase.db import connect
from ase.calculators.calculator import Calculator
from os.path import splitext
from constants import hartree_to_ev, SEED
from collections import namedtuple
from sklearn.model_selection import train_test_split

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

XyzFormat = namedtuple(
  "XyzFormat",
  ("name", "energy_patt", "string_patt", "default_unit", "parse_forces")
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


class ProvidedCalculator(Calculator):
  """
  A simple calculator which just returns the provided energy and forces.
  """

  implemented_properties = ["energy", "forces"]

  def __init__(self, atoms=None):
    """
    Initialization method.

    Args:
      atoms: an optional `ase.Atoms` object to which the calculator will be
        attached.

    """
    Calculator.__init__(self, label="provided", atoms=atoms)

  def set_atoms(self, atoms):
    """
    Set the attached `ase.Atoms` object.
    """
    self.atoms = atoms

  def calculate(self, atoms=None, properties=None, system_changes=None):
    """
    Set the calculation results.
    """
    super(ProvidedCalculator, self).calculate(atoms, properties=properties,
                                              system_changes=system_changes)
    self.results = {
      'energy': self.atoms.info.get('provided_energy', 0.0),
      'forces': self.atoms.info.get('provided_forces',
                                    np.zeros((len(self.atoms), 3)))
    }


def xyz_to_db(xyzfile, num_examples, xyz_format='xyz', verbose=True,
              unit_to_ev=None):
  """
  Convert the xyz file to an `ase.db.core.Database`.

  Args:
    xyzfile: a `str` as the file to parse.
    num_examples: a `int` as the maximum number of examples to parse.
    xyz_format: a `str` representing the format of the given xyz file.
    verbose: a `bool` indicating whether we should log the parsing progress.
    unit_to_ev: a `float` as the unit for converting energies to eV. Defaults
      to None so that default units will be used.

  Returns:
    db: an `ase.db.core.Database`.

  """
  formatter = _get_regex_patt_and_unit(xyz_format)
  assert isinstance(formatter, XyzFormat)

  name = splitext(xyzfile)[0] + ".db"
  unit = unit_to_ev or formatter.default_unit
  parse_forces = formatter.parse_forces
  count = 0
  ai = 0
  natoms = 0
  stage = 0
  atoms = None

  db = connect(name=name)
  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")
  with open(xyzfile) as f:
    for line in f:
      if count == num_examples:
        break
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          natoms = int(l)
          atoms = Atoms(calculator=ProvidedCalculator())
          if parse_forces:
            atoms.info['provided_forces'] = np.zeros((natoms, 3))
          stage += 1
      elif stage == 1:
        m = formatter.energy_patt.search(l)
        if m:
          if xyz_format.lower() == 'extxyz':
            energy = float(m.group(3)) * unit
          elif xyz_format.lower() == 'grendel':
            energy = float(m.group(2)) * unit
            atoms.set_cell(
              np.reshape([float(x) for x in m.group(1).split()], (3, 3)))
            atoms.set_pbc(
              [True if x == "T" else False for x in m.group(3).split()])
          else:
            energy = float(m.group(1)) * unit
          atoms.info['provided_energy'] = energy
          stage += 1
      elif stage == 2:
        m = formatter.string_patt.search(l)
        if m:
          atoms.append(Atom(symbol=m.group(1),
                            position=[float(v) for v in m.groups()[1:4]]))
          if parse_forces:
            atoms.info['provided_forces'][ai, :] = [float(v)
                                                    for v in m.groups()[4:7]]
          ai += 1
          if ai == natoms:
            atoms.calc.calculate()
            db.write(atoms)
            ai = 0
            stage = 0
            count += 1
            if verbose and count % 1000 == 0:
              sys.stdout.write(
                "\rProgress: %7d  /  %7d" % (count, num_examples))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

    return db


class Database:
  """
  A manager class for manipulating the `ase.db.core.Database`.
  """

  def __init__(self, db):
    """
    Initialization method.

    Args:
      db: a `ase.db.core.Database`

    """
    self._db = db
    self._random_state = SEED
    self._energy_range = None
    self._splitted = False
    self._training_ids = None
    self._testing_ids = None

  def __len__(self):
    """
    Return the total number of examples stored in this database.
    """
    return len(self._db)

  def __getitem__(self, ind):
    """
    Get one or more structures.

    Args:
      ind: an `int` or a list of `int` as the zero-based id(s) to select.

    Returns:
      sel: an `ase.Atoms` or a list of `ase.Atoms`.

    """
    if isinstance(ind, int):
      sel = self._db.get_atoms('id={}'.format(ind + 1))
    elif isinstance(ind, (list, tuple, np.ndarray)):
      self._db.update(list(ind), selected=True)
      sel = [self._get_atoms(row) for row in self._db.select(selected=True)]
      self._db.update(list(ind), selected=False)
    else:
      raise ValueError('')
    return sel

  @staticmethod
  def _get_atoms(row):
    """
    Convert the database row to `ase.Atoms` while keeping the info dict.

    Args:
      row: an `ase.db.row.AtomsRow`.

    Returns:
      atoms: an `ase.Atoms` object representing a structure.

    """
    atoms = row.toatoms()
    atoms.info.update(row.key_value_pairs)
    return atoms

  @property
  def num_examples(self):
    """
    Return the total number of examples stored in this database.
    """
    return len(self._db)

  @property
  def ids_of_training_examples(self):
    """
    Return the ids for all training examples.
    """
    return self._training_ids

  @property
  def ids_of_testing_examples(self):
    """
    Return the ids for all testing examples.
    """
    return self._testing_ids

  @property
  def energy_range(self):
    """
    Return the energy range of this database.
    """
    if self._energy_range is None:
      self._get_energy_range()
    return self._energy_range

  def _get_energy_range(self):
    """
    Determine the energy range.
    """
    y_min = np.inf
    y_max = -np.inf
    for row in self._db.select(list(range(len(self)))):
      y_min = min(row.energy, y_min)
      y_max = max(row.energy, y_max)
    self._energy_range = (y_min, y_max)

  def split(self, test_size=0.2, random_state=None):
    """
    Split this database into training set and testing set.

    Args:
      test_size: a `float` or `int`. If float, should be between 0.0 and 1.0 and
        represent the proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples.
      random_state: a `int` as the pseudo-random number generator state used for
        random sampling.

    """
    random_state = random_state or self._random_state
    indices = range(len(self))
    training_ids, testing_ids = train_test_split(
      indices, test_size=test_size, random_state=random_state)
    self._db.update(training_ids, for_training=True)
    self._db.update(testing_ids, for_training=False)
    self._random_state = random_state
    self._splitted = True
    self._training_ids = training_ids
    self._testing_ids = testing_ids

  def examples(self, for_training=True):
    """
    A set-like object providing a view on `ase.Atoms` of this database.

    Args:
      for_training: a `bool` indicating whether should we view on training
        examples or not.

    Yields:
      atoms: an `ase.Atoms` object.

    """
    if not self._splitted:
      self.split()
    for row in self._db.select(for_training=for_training):
      yield self._get_atoms(row)

  @classmethod
  def from_xyz(cls, xyzfile, num_examples, xyz_format='xyz', verbose=True,
               unit_to_ev=None):
    """
    Initialize a `Database` from a xyz file.

    Args:
      xyzfile: a `str` as the file to parse.
      num_examples: a `int` as the maximum number of examples to parse.
      xyz_format: a `str` representing the format of the given xyz file.
      verbose: a `bool` indicating whether we should log the parsing progress.
      unit_to_ev: a `float` as the unit for converting energies to eV. Defaults
        to None so that default units will be used.

    Returns:
      db: a `Database`.

    """
    return cls(xyz_to_db(xyzfile,
                         num_examples=num_examples,
                         xyz_format=xyz_format,
                         verbose=verbose,
                         unit_to_ev=unit_to_ev))

  @classmethod
  def from_file(cls, filename):
    """
    Initialize a `Database` from a db.

    Args:
      filename: a `str` as the file to load.

    Returns:
      db: a `Database`.

    """
    with connect(filename) as db:
      return cls(db)
