#!coding=utf-8
"""
The global constants.
"""
from __future__ import print_function, absolute_import

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

# Set the random state
SEED = 218

# Hartree to kcal/mol
hartree_to_kcal = 627.509

# Hartree to eV
hartree_to_ev = 27.211

# a.u to angstroms
au_to_angstrom = 0.52917721092

# The pyykko radius of each element.
pyykko = {
  'Ac': 1.86, 'Ag': 1.28, 'Al': 1.26, 'Am': 1.66, 'Ar': 0.96, 'As': 1.21,
  'At': 1.47, 'Au': 1.24, 'B': 0.85, 'Ba': 1.96, 'Be': 1.02, 'Bh': 1.41,
  'Bi': 1.51, 'Bk': 1.68, 'Br': 1.14, 'C': 0.75, 'Ca': 1.71, 'Cd': 1.36,
  'Ce': 1.63, 'Cf': 1.68, 'Cl': 0.99, 'Cm': 1.66, 'Co': 1.11, 'Cr': 1.22,
  'Cs': 2.32, 'Cu': 1.12, 'Db': 1.49, 'Ds': 1.28, 'Dy': 1.67, 'Er': 1.65,
  'Es': 1.65, 'Eu': 1.68, 'F': 0.64, 'Fe': 1.16, 'Fm': 1.67, 'Fr': 2.23,
  'Ga': 1.24, 'Gd': 1.69, 'Ge': 1.21, 'H': 0.32, 'He': 0.46, 'Hf': 1.52,
  'Hg': 1.33, 'Ho': 1.66, 'Hs': 1.34, 'I': 1.33, 'In': 1.42, 'Ir': 1.22,
  'K': 1.96, 'Kr': 1.17, 'La': 1.8, 'Li': 1.33, 'Lu': 1.62, 'Md': 1.73,
  'Mg': 1.39, 'Mn': 1.19, 'Mo': 1.38, 'Mt': 1.29, 'N': 0.71, 'Na': 1.55,
  'Nb': 1.47, 'Nd': 1.74, 'Ne': 0.67, 'Ni': 1.1, 'No': 1.76, 'Np': 1.71,
  'O': 0.63, 'Os': 1.29, 'P': 1.11, 'Pa': 1.69, 'Pb': 1.44, 'Pd': 1.2,
  'Pm': 1.73, 'Po': 1.45, 'Pr': 1.76, 'Pt': 1.23, 'Pu': 1.72, 'Ra': 2.01,
  'Rb': 2.1, 'Re': 1.31, 'Rf': 1.57, 'Rh': 1.25, 'Rn': 1.42, 'Ru': 1.25,
  'S': 1.03, 'Sb': 1.4, 'Sc': 1.48, 'Se': 1.16, 'Sg': 1.43, 'Si': 1.16,
  'Sm': 1.72, 'Sn': 1.4, 'Sr': 1.85, 'Ta': 1.46, 'Tb': 1.68, 'Tc': 1.28,
  'Te': 1.36, 'Th': 1.75, 'Ti': 1.36, 'Tl': 1.44, 'Tm': 1.64, 'U': 1.7,
  'V': 1.34, 'W': 1.37, 'Xe': 1.31, 'Y': 1.63, 'Yb': 1.7,
  'Zn': 1.18, 'Zr': 1.54,
  # X represents virtual atoms, the covalent radius is the half of the LJ
  # distance.
  'X': 0.5 * 1.122462048309373,
}

# the ghost atom
GHOST = 'X'

# Constants describing the training process.

# The decay to use for the moving average of variable weights.
VARIABLE_MOVING_AVERAGE_DECAY = 0.9999

# The decay to use for the moving average of losses.
LOSS_MOVING_AVERAGE_DECAY = 0.9

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


class KcnnGraphKeys:
  """
  Custom graph collections for kCON.

  * `ENERGY_VARIABLES`: variables that should be updated during energy training.
  * `FORCES_VARIABLES`: variables that should be updated during forces training.

  """
  ENERGY_VARIABLES = 'energy'
  FORCES_VARIABLES = 'forces'
