#!coding=utf-8
"""
This module provides an `ase.calculator.Calculator` wrapper of `KcnnPredictor`.
"""
from __future__ import print_function, absolute_import

from ase.calculators.calculator import Calculator
from monty.design_patterns import cached_class
from predictor import KcnnPredictor

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@cached_class
class Kcnn(Calculator):
  """
  An `ase.calculator.Calculator` wrapper of the `KcnnPredictor`.
  """

  implemented_properties = ['energy', 'kbody', '1body', 'atomic']
  default_parameters = {}
  nolabel = True

  def __init__(self, graph_model_path):
    """
    Initialization method.

    Args:
      graph_model_path: a `str` as the model file to read.

    """
    Calculator.__init__(self)
    self._clf = KcnnPredictor(graph_model_path)

  def calculate(self, atoms=None, properties=('energy', 'atomic'), *args):
    """
    Calculate the total energy and other properties (1body, kbody, atomic).

    Args:
      atoms: an `ase.Atoms` to calculate.
      properties: a list of `str` as the properties to calculate. Available
        options are: 'energy', 'atomic', '1body' and 'kbody'.

    """
    Calculator.calculate(self, atoms, properties, *args)

    if len(properties) == 1 and 'energy' in properties:
      y_total = self._clf.predict_total_energy(self.atoms)
      self.results = {'energy': float(y_total)}

    else:
      y_total, y_1body, y_atomic, y_kbody = self._clf.predict(atoms)
      self.results = {'energy': float(y_total),
                      '1body': y_1body,
                      'kbody': y_kbody,
                      'atomic': y_atomic}

  def get_atomic_energy(self, atoms=None):
    """
    A convenient function to get the atomic energies.
    """
    return self.get_property('atomic', atoms=atoms)

  def get_kbody_energy(self, atoms=None):
    """
    A convenient function to get the k-body energy contributions.
    """
    return self.get_property('kbody', atoms=atoms)
