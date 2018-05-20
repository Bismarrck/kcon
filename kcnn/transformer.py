# coding=utf-8
"""
This script is used to transform atomic coordinates to input features and then
save them in tfrecords files.
"""

from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import json
import sys
import time
from collections import Counter, namedtuple
from itertools import combinations, product, repeat, chain
from functools import partial
from os.path import basename, dirname, join, splitext
from ase.atoms import Atoms
from scipy.misc import comb
from sklearn.metrics import pairwise_distances
from tensorflow.python.training.training import Features, Example
from constants import pyykko, GHOST, LJR
from utils import get_atoms_from_kbody_term, safe_divide, compute_n_from_cnk
from utils import Gauss

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# TODO: fix the multi-counting problem when the number of ghost atoms >= 2.
# TODO: implement `compute_atomic_energies` for `FixedLenMultiTransformer`.

FLAGS = tf.app.flags.FLAGS

# Important: this `_safe_log` must be very very small. The previous e^-6 is
# large enough to cause significant numeric errors.
_safe_log = np.e**(-20)


"""
A data structure for storing transformed features and auxiliary parameters.
"""
KcnnSample = namedtuple("KcnnSample", (
  "features",
  "split_dims",
  "binary_weights",
  "occurs",
  "coefficients",
  "indexing",
  "compress_stats"
))


def get_formula(species):
  """
  Return the molecular formula given a list of atomic species.
  """
  return "".join(species)


def _compute_lr_weights(coef, y, num_real_atom_types, factor=1.0):
  """
  Solve the linear equation system of Ax = b.
  
  Args:
    coef: a `float` array of shape `[num_examples, num_atom_types]`.
    y: a `float` array of shape `[num_examples, ]`.
    num_real_atom_types: an `int` as the number of atom types excluding the
      ghost atoms.
    factor: a `float` as a scaling factor for the weights.

  Returns:
    x: a `float` array of shape `[num_atom_types, ]` as the solution.

  """
  rank = np.linalg.matrix_rank(coef[:, :num_real_atom_types])
  diff = num_real_atom_types - rank

  # The coef matrix is full rank. So the linear equation system can be solved.
  if diff == 0:
    x = np.negative(np.dot(np.linalg.pinv(coef), y))

  # The rank is 1, so all structures have the same stoichiometry. Then all types
  # of atoms can be treated equally.
  elif rank == 1:
    x = np.negative(np.mean(y / coef[:, :num_real_atom_types].sum(axis=1)))

  else:
    raise ValueError(
      "Coefficients matrix rank {} of {} is not supported!".format(
        rank, num_real_atom_types))

  return x * factor


def _get_pyykko_bonds_matrix(species, factor=1.0, flatten=True, lj=False):
  """
  Return the pyykko-bonds matrix given a list of atomic symbols.

  Args:
    species: a `List[str]` as the atomic symbols.
    factor: a `float` as the normalization factor.
    flatten: a `bool` indicating whether the bonds matrix is flatten or not.
    lj: a `bool`. If True, all atoms will be treated as ideal LJ atoms.

  Returns:
    bonds: the bonds matrix (or vector if `flatten` is True).

  """
  if not lj:
    rr = np.asarray([pyykko[specie] for specie in species])[:, np.newaxis]
  else:
    rr = np.ones((len(species), 1)) * LJR
  lmat = np.multiply(factor, rr + rr.T)
  if flatten:
    return lmat.flatten()
  else:
    return lmat


def get_kbody_terms_from_species(species, k_max):
  """
  Return the k-body terms given the chemical symbols and `many_body_k`.

  Args:
    species: a `list` of `str` as the chemical symbols.
    k_max: a `int` as the maximum k-body terms that we should consider.

  Returns:
    kbody_terms: a `list` of `str` as the k-body terms.

  """
  return sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, k_max)])))


def _get_num_force_entries(n, k_max):
  """
  Return the number of entries per force component.

  Args:
    n: an `int` as the maximum number of 'real' atoms in a structure.
    k_max: an `int` as the maximum k.

  Returns:
    num_entries: an `int` as the number of entries per force component.

  """
  return int(np.array([comb(n, k) * comb(k, 2) * 2 / n
                       for k in range(2, k_max + 1)]).sum())


def exponential_norm(x, unit=1.0, order=1):
  """
  Normalize the inputs `x` with the exponential function:
    f(x) = exp(-x / unit)

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    unit: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).
    order: a `int` as the exponential order. If `order` is 0, the inputs will 
      not be scaled by `factor`.

  Returns:
    scaled: the scaled unitless inputs.

  """
  if order == 0:
    return np.exp(-x)
  else:
    return np.exp(-(x / unit) ** order)


def exponential_gauss(x, unit=1.0):
  """
  Normalize the inputs `x` with the mixed functions:
    f(x) = exp(-x / unit),               if x / unit >= 1.0
    f(x) = gaussian(1.0, 0.2)(x / unit), else

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    unit: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).

  Returns:
    scaled: the scaled unitless inputs.

  """
  g = Gauss(1.0, 0.2)
  r = x / unit
  scale = np.exp(-1.0) / g(1.0)
  if hasattr(r, '__getitem__'):
    left = np.where(r < 1.0)[0]
    y = np.exp(-r)
    y[left] = g(r[left]) * scale
    return y
  else:
    if r < 1.0:
      return g(r) * scale
    else:
      return np.exp(-r)


def lj_norm(x, unit=1.0):
  """
  Normalize the inputs `x` with the Lennard-Jones function.

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    unit: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).

  Returns:
    scaled: the scaled unitless inputs.

  """
  rm = x / unit
  left = np.where(rm < 1.0)[0]
  rp = rm ** 4
  rp[left] = rm[left] ** 2
  y = safe_divide(1.0, rp)
  return np.clip(-(y ** 2 - 2.0 * y), 0.0, 1.0)


def morse(x, unit=1.0, order=1):
  """
  Normalize the inputs `x` with the Morse function.

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    unit: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).
    order: a `float` or an `int`.

  Returns:
    scaled: the scaled unitless inputs.

  """
  alpha = -1.0 * float(order)
  return (1.0 - np.exp(alpha * (x - unit)))**2


def _bytes_feature(value):
  """
  Convert the `value` to Protobuf bytes.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """
  Convert the `value` to Protobuf float32.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class Transformer:
  """
  This class is used to transform atomic coordinates to input feature matrix.
  """

  def __init__(self, species, k_max=3, kbody_terms=None, split_dims=None,
               norm='exp', norm_order=1, periodic=False, atomic_forces=False,
               lj=False, cutoff=None):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      kbody_terms: a `List[str]` as the k-body terms.
      split_dims: a `List[int]` as the dimensions for splitting inputs. If this
        is given, the `kbody_terms` must also be set and their lengths should be
        equal.
      norm: a `str` specifying the normalization function to use. Defaults to
        `exp`, `lj`, `exp+g` and `morse` are also supported.
      norm_order: a `int` as the order for normalizing interatomic distances.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.
      atomic_forces: a `bool` indicating whether the atomic forces derivation is
        enabled or not.
      lj: a `bool` indicating that this transformer targets on LJ systems.
      cutoff: a `float` as the cutoff.

    """
    if split_dims is not None:
      assert len(split_dims) == len(kbody_terms)

    kbody_terms = kbody_terms or get_kbody_terms_from_species(species, k_max)
    num_ghosts = self._get_num_ghosts(species, k_max)
    mapping, selections = self._get_mapping(species, kbody_terms)

    # Internal initialization.
    offsets, real_dim, kbody_sizes = [0], 0, []
    if split_dims is None:
      # If `split_dims` is not given, we shall construct it by our own.
      # `real_dim` should be always not smaller than `sum(kbody_sizes)` because
      # every excluded k-body term is represented by a single row vector of all
      # zeros.
      for kbody_term in kbody_terms:
        if kbody_term in mapping:
          size = mapping[kbody_term].shape[1]
        else:
          size = 0
        real_dim += max(size, 1)
        kbody_sizes.append(size)
        offsets.append(real_dim)
      split_dims = np.diff(offsets).tolist()
      # Here we must use `sum(kbody_sizes)` but not `offsets[-1]` because when
      # `split_dims` is not set while `kbody_terms` is given, `offsets[-1]` is
      # not equal to C(N_max, k).
      n = compute_n_from_cnk(sum(kbody_sizes), k_max)

    else:
      offsets = [0] + np.cumsum(split_dims).tolist()
      real_dim = offsets[-1]
      kbody_sizes = []
      for kbody_term in kbody_terms:
        if kbody_term in mapping:
          size = mapping[kbody_term].shape[1]
        else:
          size = 0
        kbody_sizes.append(size)
      n = compute_n_from_cnk(offsets[-1], k_max)

    # Initialize internal variables.
    self._lj = lj
    self._k_max = k_max
    self._kbody_terms = kbody_terms
    self._offsets = offsets
    self._kbody_sizes = kbody_sizes
    self._species = species
    self._mapping = mapping
    self._selections = selections
    self._split_dims = split_dims
    self._ck2 = int(comb(k_max, 2, exact=True))
    self._cond_sort = self._get_conditional_sorting_indices(kbody_terms)
    self._cmatrix = _get_pyykko_bonds_matrix(species, lj=lj)
    self._num_ghosts = num_ghosts
    self._periodic = periodic
    self._real_dim = real_dim
    self._binary_weights = self._get_binary_weights()
    self._atomic_forces = atomic_forces
    self._indexing_matrix = None
    self._num_real = n - self._num_ghosts
    self._num_f_components = 3 * self._num_real
    self._num_entries = _get_num_force_entries(self._num_real, self._k_max)
    self._norm = norm
    if norm.lower() == 'exp':
      self._norm_fn = partial(exponential_norm, order=norm_order)
    elif norm.lower() == 'lj':
      self._norm_fn = lj_norm
    elif norm.lower() == 'exp+g':
      self._norm_fn = exponential_gauss
    elif norm.lower() == 'morse':
      self._norm_fn = partial(morse, order=norm_order)
    else:
      raise ValueError("Unsupported normalizing function: {}".format(norm))
    self._cutoff = cutoff or np.inf
    self._cutoff_table = self._get_cutoff_table()

  @property
  def species(self):
    """
    Return the species of this transformer excluding all ghosts.
    """
    return [symbol for symbol in self._species if symbol != GHOST]

  @property
  def shape(self):
    """
    Return the shape of the transformed input feature matrix.
    """
    return self._real_dim, self._ck2

  @property
  def ck2(self):
    """
    Return the value of C(k,2) for this transformer.
    """
    return self._ck2

  @property
  def k_max(self):
    """
    Return the maximum order for the many-body expansion.
    """
    return self._k_max

  @property
  def split_dims(self):
    """
    Return the dims for spliting the inputs.
    """
    return self._split_dims

  @property
  def kbody_terms(self):
    """
    Return the kbody terms for this transformer. 
    """
    return self._kbody_terms

  @property
  def kbody_sizes(self):
    """
    Return the real sizes of each kbody term of this transformer. Typically this
    is equal to `split_dims` but when `kbody_terms` is manually set, this may be 
    different.
    """
    return self._kbody_sizes

  @property
  def binary_weights(self):
    """
    Return the binary weights for the all k-body contribs.
    """
    return self._binary_weights

  @property
  def kbody_selections(self):
    """
    Return the kbody selections.
    """
    return self._selections

  @property
  def num_ghosts(self):
    """
    Return the number of ghosts atoms.
    """
    return self._num_ghosts

  @property
  def is_periodic(self):
    """
    Return True if this transformer is used for periodic structures.
    """
    return self._periodic

  @property
  def support_atomic_forces(self):
    """
    Return True if atomic forces derivation is enabled.
    """
    return self._atomic_forces

  @property
  def num_force_components(self):
    """
    Return the total number of force components.
    """
    return self._num_f_components

  @property
  def num_entries_per_component(self):
    """
    Return the number of entries per force component.
    """
    return self._num_entries

  @property
  def is_lj(self):
    """
    Return True if this transformer shall be used for LJ systems.
    """
    return self._lj

  @property
  def cutoff(self):
    """
    Return the cutoff.
    """
    return self._cutoff

  def get_bond_types(self):
    """
    Return the ordered bond types for each k-body term.
    """
    bonds = {}
    for kbody_term in self._kbody_terms:
      atoms = get_atoms_from_kbody_term(kbody_term)
      bonds[kbody_term] = ["-".join(ab) for ab in combinations(atoms, r=2)]
    return bonds

  def _get_cutoff_table(self):
    """
    Return the cutoff table.
    """
    table = np.zeros((self._real_dim, self._ck2), dtype=np.float32)
    cutoff = self._cutoff
    if cutoff < np.inf:
      thres = self._norm_fn(cutoff)
    else:
      thres = 0.0
    for i, kbody_term in enumerate(self._kbody_terms):
      istart, istop = self._offsets[i], self._offsets[i + 1]
      if GHOST in kbody_term:
        table[istart: istop, 0] = thres
      else:
        table[istart: istop, :] = thres
    return table

  @staticmethod
  def _get_num_ghosts(species, many_body_k):
    """
    Return and check the number of ghost atoms.

    Args:
      species: a `list` of `str` as the chemical symbols.
      many_body_k: a `int` as the maximum k-body terms that we should consider.

    Returns:
      num_ghosts: a `int` as the number of ghost atoms.

    """
    num_ghosts = list(species).count(GHOST)
    if num_ghosts != 0 and (num_ghosts > 2 or many_body_k - num_ghosts != 2):
      raise ValueError("The number of ghosts is wrong!")
    return num_ghosts

  @staticmethod
  def _get_mapping(species, kbody_terms):
    """
    Build the mapping from interatomic distance matrix of shape `[N, N]` to the
    input feature matrix of shape `[C(N, k), C(k, 2)]`.

    Args:
      species: a `list` of `str` as the ordered atomic symbols.
      kbody_terms: a `list` of `str` as the ordered k-body terms.

    Returns:
      mapping: a `Dict[str, Array]` as the mapping from the N-by-N interatomic
        distance matrix to the input feature matrix for each k-body term.
      selection: a `Dict[str, List[List[int]]]` as the indices of the k-atoms
        selections for each k-body term.

    """
    natoms = len(species)
    mapping = {}
    selections = {}

    # Determine the indices of each type of atom and store them in a dict.
    atom_index = {}
    for i in range(len(species)):
      atom = species[i]
      atom_index[atom] = atom_index.get(atom, []) + [i]

    for kbody_term in kbody_terms:
      # Extract atoms from this k-body term
      atoms = get_atoms_from_kbody_term(kbody_term)
      # Count the occurances of the atoms.
      counter = Counter(atoms)
      # If an atom appears more in the k-body term, we should discard this
      # k-body term. For example the `CH4` molecule can not have `CCC` or `CCH`
      # interactions.
      if any([counter[e] > len(atom_index.get(e, [])) for e in atoms]):
        continue
      # ck2 is the number of bond types in this k-body term.
      ck2 = int(comb(len(atoms), 2, exact=True))
      # Sort the atoms
      sorted_atoms = sorted(counter.keys())
      # Build up the k-atoms selection candidates. For each type of atom we draw
      # N times where N is equal to `counter[atom]`. Thus, the candidate list
      # can be constructed:
      # [[[1, 2], [1, 3], [1, 4], ...], [[8], [9], [10], ...]]
      # The length of the candidates is equal to the number of atom types.
      k_atoms_candidates = [
        [list(o) for o in combinations(atom_index[e], counter[e])]
        for e in sorted_atoms
      ]
      # Build up the k-atoms selections. First, we get the `product` (See Python
      # official document for more info), eg [[1, 2], [8]]. Then `chain` it to
      # get flatten lists, eg [[1, 2, 8]].
      k_atoms_selections = [list(chain(*o)) for o in
                            product(*k_atoms_candidates)]
      selections[kbody_term] = k_atoms_selections
      # cnk is the number of k-atoms selections.
      cnk = len(k_atoms_selections)
      # Construct the mapping from the interatomic distance matrix to the input
      # matrix. This procedure can greatly increase the transformation speed.
      # The basic idea is to fill the input feature matrix with broadcasting.
      # The N-by-N interatomic distance matrix is flatten to 1D vector. Then we
      # can fill the matrix like this:
      #   feature_matrix[:, col] = flatten_dist[[1,2,8,10,9,2,1,1]]
      mapping[kbody_term] = np.zeros((ck2, cnk), dtype=int)
      for i in range(cnk):
        for j, (vi, vj) in enumerate(combinations(k_atoms_selections[i], 2)):
          mapping[kbody_term][j, i] = vi * natoms + vj
    return mapping, selections

  @staticmethod
  def _get_conditional_sorting_indices(kbody_terms):
    """
    Generate the indices of the columns for the conditional sorting scheme.

    Args:
      kbody_terms: a `List[str]` as the ordered k-body terms.

    Returns:
      indices: a `dict` of indices for sorting along the last axis of the input
        features.

    """
    indices = {}
    for kbody_term in kbody_terms:
      # Extract atoms from this k-body term
      atoms = get_atoms_from_kbody_term(kbody_term)
      # All possible bonds from the given atom types
      bonds = list(combinations(atoms, r=2))
      n = len(bonds)
      counter = Counter(bonds)
      # If the bonds are unique, there is no need to sort because columns of the
      # formed feature matrix will not be interchangable.
      if max(counter.values()) == 1:
        continue
      # Determine the indices of duplicated bonds.
      indices[kbody_term] = []
      for bond, times in counter.items():
        if times > 1:
          indices[kbody_term].append([i for i in range(n) if bonds[i] == bond])
    return indices

  def _get_binary_weights(self):
    """
    Return the binary weights.
    """
    weights = np.zeros(self._real_dim, dtype=np.float32)
    offsets = self._offsets
    for i in range(len(self._split_dims)):
      weights[offsets[i]: offsets[i] + self._kbody_sizes[i]] = 1.0
    return weights

  def _get_coords(self, atoms):
    """
    Return the N-by-3 coordinates matrix for the given `ase.Atoms`.

    Notes:
      Auxiliary vectors may be appended if `num_ghosts` is non-zero.

    """
    if self._num_ghosts > 0:
      # Append `num_ghosts` rows of zeros to the positions. We can not directly
      # use `inf` because `pairwise_distances` and `get_all_distances` do not
      # support `inf`.
      aux_vecs = np.zeros((self._num_ghosts, 3))
      coords = np.vstack((atoms.get_positions(), aux_vecs))
    else:
      coords = atoms.get_positions()
    return coords

  def _get_interatomic_distances(self, coords, cell, pbc):
    """
    Return the interatomic distances matrix and its associated coordinates
    differences matrices.

    Returns:
      dists: a `float32` array of shape `[N, N]` where N is the number of atoms
        as the interatomic distances matrix.
      delta: a `float32` array of shape `[N, N, 3]`. The last axis represents
        the X, Y, Z directions. `delta[:, :, 0]` is the pairwise distances of
        all X coordinates.

    """
    delta = None

    if not self.is_periodic:
      dists = pairwise_distances(coords)
      if self._atomic_forces:
        # Compute dx_ij, dy_ij and dz_ij. `sklearn.metrics.pairwise_distances`
        # cannot be used here because it will return absolute values.
        ndim = len(dists)
        delta = np.zeros((ndim, ndim, 3))

        # Based on my tests, the vector subtraction used here should be:
        # r_{ij} = r_{i} - r_{j}
        for i in range(ndim):
          for j in range(i + 1, ndim):
            delta[i, j, :] = coords[i] - coords[j]  # r_ij
            delta[j, i, :] = coords[j] - coords[i]  # r_ji
    else:
      atoms = Atoms(
        symbols=self._species,
        positions=coords,
        cell=cell,
        pbc=pbc
      )
      dists = atoms.get_all_distances(mic=True)
      if self._atomic_forces:
        raise ValueError("Atomic forces for periodic structures are not "
                         "supported yet!")
      del atoms

    # Manually set the distances between ghost atoms and real atoms to inf.
    if self._num_ghosts > 0:
      dists[:, -self._num_ghosts:] = np.inf
      dists[-self._num_ghosts:, :] = np.inf

      # Set the delta differences to 0.
      if self._atomic_forces:
        delta[:, -self._num_ghosts:, :] = 0.0
        delta[-self._num_ghosts:, :, :] = 0.0

    return dists, delta

  def _assign(self, dists, delta, features=None):
    """
    Assign the normalized distances to the input feature matrix and build the
    auxiliary matrices.

    Args:
      dists: a `float32` array of shape `[N**2, ]` as the scaled flatten
        interatomic distances matrix.
      delta: a `float32` array of shape `[N**2, 3]`.
      features: a 2D `float32` array or None as the location into which the
        result is stored. If not provided, a new array will be allocated.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.
      cr: a `float32` array of shape `self.shape` as the covalent radii for
        corresponding entries.
      dr: a `floar32` array of shape `[self.shape[0], 6 * self.shape[1]]` as the
        corresponding differences of coordinates.

    """
    if features is None:
      features = np.zeros((self._real_dim, self._ck2))
    elif features.shape != self.shape:
      raise ValueError("The shape should be {}".format(self.shape))

    if self._atomic_forces:
      cr = np.zeros_like(features)
      dr = np.zeros((self._real_dim, self._ck2 * 6))
    else:
      cr = None
      dr = None

    half = self._ck2 * 3
    step = self._ck2
    zero = 0

    for i, kbody_term in enumerate(self._kbody_terms):
      if kbody_term not in self._mapping:
        continue
      # The index matrix was transposed because typically C(N, k) >> C(k, 2).
      # See `_get_mapping`.
      mapping = self._mapping[kbody_term]
      istart = self._offsets[i]
      # Manually adjust the step size because the offset length may be larger if
      # `split_dims` is fixed.
      istep = min(self._offsets[i + 1] - istart, mapping.shape[1])
      istop = istart + istep
      for k in range(self._ck2):
        features[istart: istop, k] = dists[mapping[k]]
        if self._atomic_forces:
          cr[istart: istop, k] = self._cmatrix[mapping[k]]
          # x = 0, y = 1, z = 2
          for j in range(3):
            dr[istart: istop, j * step + k + zero] = +delta[mapping[k], j]
            dr[istart: istop, j * step + k + half] = -delta[mapping[k], j]

    return features, cr, dr

  def _conditionally_sort(self, features, cr, dr):
    """
    Apply the conditional sorting algorithm.

    Args:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.
      cr: a `float32` array of shape `self.shape` as the covalent radii for
        corresponding entries.
      dr: a `floar32` array of shape `[self.shape[0], 6 * self.shape[1]]` as the
        corresponding differences of coordinates.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.
      cr: a `float32` array of shape `self.shape` as the covalent radii for
        corresponding entries.
      dr: a `floar32` array of shape `[self.shape[0], 6 * self.shape[1]]` as the
        corresponding differences of coordinates.
      indexing: a `int` array of shape `[self.shape[0], self.shape[1], 2]`.

    """

    def cond_sort(_part, _orders, _shape):
      """
      Do the conditional sorting.
      """
      return _part.flatten()[_orders].reshape(_shape)

    indexing = self._get_indexing_matrix().copy()
    ck2 = self._ck2

    for i, kbody_term in enumerate(self._kbody_terms):

      if kbody_term not in self._mapping:
        continue

      for ix in self._cond_sort.get(kbody_term, []):
        z = features[self._offsets[i]: self._offsets[i + 1], ix]

        # The simple case: the derivation of atomic forces is disabled.
        if not self._atomic_forces:
          # `samples` is a 2D array, the Python advanced slicing will make the
          # returned `z` a copy but not a view. The shape of `z` is transposed.
          # So we should sort along axis 0 here!
          z.sort()
          features[self._offsets[i]: self._offsets[i + 1], ix] = z

        # When atomic forces are required, a different strategy should be used.
        # The two auxiliary matrices, covalent radii matrix and the coordinates
        # differences matrix, must all be sorted. So we get the sorting orders
        # at first.
        else:
          orders = np.argsort(z)
          istart = self._offsets[i]
          istop = self._offsets[i + 1]
          shape = z.shape
          step = shape[1]
          orders += np.tile(np.arange(0, z.size, step), (step, 1)).T

          # Sort the features
          features[istart: istop, ix] = cond_sort(z, orders, shape)

          # Sort the covalend radii
          cr[istart: istop, ix] = cond_sort(
            cr[istart: istop, ix], orders, shape)

          # Sort the indices
          for axis in range(2):
            array = indexing[istart: istop, :, axis]
            array[:, ix] = cond_sort(array[:, ix], orders, shape)
            indexing[istart: istop, :, axis] = array

          # Sort the delta coordinates: dx, dy, dz
          for axis in range(6):
            cstart = axis * ck2
            cstop = cstart + ck2
            array = dr[istart: istop, cstart: cstop]
            array[:, ix] = cond_sort(array[:, ix], orders, shape)
            dr[istart: istop, cstart: cstop] = array

    return features, cr, dr, indexing

  def _get_coef_matrix(self, z, l, d6):
    """
    Return the tiled coefficients matrix with the following equation:

    C = np.tile((z * d) / (l**2 * log(z)), (1, 6))

    Args:
      z: a `float32` array of shape `self.shape` as the input feature matrix.
      l: a `float32` array of shape `self.shape` as the covalent radii matrix.
      d6: a `float32` array of shape `[self.shape[0], self.shape[1] * 6]` as the
        differences of the coordinates.

    Returns:
      coef: a `float32` array as the coefficients matrix. The shape of `coef` is
        the same with input `d`.

    """
    if self._atomic_forces:
      # There will be zeros in `z`. Here we make sure every entry is no smaller
      # than e^-6.
      logz6 = np.tile(np.log(z.clip(min=_safe_log)), (1, 6))
      z6 = np.tile(z, (1, 6))
      l6 = np.tile(l**2, (1, 6))
      coef = safe_divide(z6 * d6, l6 * logz6)
      # There will be zeros in `l`. Here we convert all NaNs to zeros.
      return np.nan_to_num(coef)
    else:
      return None

  def _get_indexing_matrix(self):
    """
    Return the matrix for indexing the gradients matrix.
    """
    if self._indexing_matrix is None:
      cnk = self._real_dim
      ck2 = self._ck2
      # Set all entries to -1
      index_matrix = np.zeros((cnk, ck2, 2), dtype=int) - 1
      for i, kbody_term in enumerate(self._kbody_terms):
        if kbody_term not in self._selections:
          continue
        selections = self._selections[kbody_term]
        offset = self._offsets[i]
        for j, selection in enumerate(selections):
          for l, (a, b) in enumerate(combinations(selection, r=2)):
            index_matrix[offset + j, l] = a, b
      self._indexing_matrix = index_matrix
    return self._indexing_matrix

  def _transform_indexing_matrix(self, indexing):
    """
    Transform the conditionally sorted indexing matrix.

    Args:
      indexing: an `int` array of shape `[self.shape[0], self.shape[1], 2]` as
        the indexing matrix.

    Returns:
      positions: an `int` array of shape `[3N, C(N, k) * C(k, 2) * 2 / N]` as
        the positions of the entries for each atomic force component.

    """
    if not self._atomic_forces:
      return None

    cnk = indexing.shape[0]
    ck2 = indexing.shape[1]
    positions = np.zeros((self._num_f_components, self._num_entries), dtype=int)
    loc = np.zeros((self._num_real, ), dtype=int)
    position = 0
    half = ck2 * 3
    zero = 0
    imax = len(self.species)

    # The index should start from 1. 0 will be used as the virtual index
    # corresponding to zero contribution.
    start = 1

    for i in range(cnk):
      if indexing[i].min() >= 0:
        for j in range(ck2):
          a, b = indexing[i, j, :]
          # The contributions from Atom-Ghost pairs should be ignored.
          if a >= imax or b >= imax:
            continue
          ax = a * 3 + 0
          ay = a * 3 + 1
          az = a * 3 + 2
          bx = b * 3 + 0
          by = b * 3 + 1
          bz = b * 3 + 2
          positions[ax, loc[a]] = position + 0 * ck2 + j + zero + start
          positions[ay, loc[a]] = position + 1 * ck2 + j + zero + start
          positions[az, loc[a]] = position + 2 * ck2 + j + zero + start
          loc[a] += 1
          positions[bx, loc[b]] = position + 0 * ck2 + j + half + start
          positions[by, loc[b]] = position + 1 * ck2 + j + half + start
          positions[bz, loc[b]] = position + 2 * ck2 + j + half + start
          loc[b] += 1

      position += 6 * ck2

    return positions

  def compress(self, features):
    """
    Apply the soft compressing algorithm. The compression is implemented by
    adjusting the binary weights.

    Args:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.

    Returns:
      weights: a `float32` array as the updated binary weights.
      counter: a `dict` to count the number of kept contribs for each k-body
        term. This may be an empty `dict` indicating all contribs are kept.

    """
    if self._cutoff == np.inf:
      return self._binary_weights, {}

    else:
      results = np.sum(features >= self._cutoff_table, axis=1, dtype=int)
      weights = np.ones_like(self._binary_weights)
      weights[results < 3] = 0.0
      counter = {}
      for i, kbody_term in enumerate(self._kbody_terms):
        istart, istop = self._offsets[i], self._offsets[i + 1]
        counter[kbody_term] = np.sum(results[istart: istop] == 3)
      return weights, counter

  def transform(self, atoms, features=None):
    """
    Transform the given `ase.Atoms` object to an input feature matrix.

    Args:
      atoms: an `ase.Atoms` object.
      features: a 2D `float32` array or None as the location into which the
        result is stored. If not provided, a new array will be allocated.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.
      coef: a `float32` array as the coefficients matrix. The shape of `coef` is
        `[self.shape[0], self.shape[1] * 6]`.
      indexing: an `int` array of shape `[3N, C(N, k) * C(k, 2) * 2 / N]` as the
        indices of the entries for each atomic force component.

    """

    # Get the coordinates matrix.
    coords = self._get_coords(atoms)

    # Compute the interatomic distances. For non-periodic molecules we use the
    # faster method `pairwise_distances`.
    dists, delta = self._get_interatomic_distances(
      coords,
      cell=atoms.get_cell(),
      pbc=atoms.get_pbc()
    )

    # Normalize the interatomic distances with the exponential function so that
    # shorter bonds have larger normalized weights.
    dists = dists.flatten()
    if delta is not None:
      delta = delta.reshape((-1, 3))
    norm_dists = self._norm_fn(dists, unit=self._cmatrix.flatten())

    # Assign the normalized distances to the input feature matrix.
    features, cr, dr = self._assign(norm_dists, delta, features=features)

    # Apply the conditional sorting algorithm
    features, cr, dr, indexing = self._conditionally_sort(features, cr, dr)

    # Get coefficients matrix for computing atomic forces.
    coef = self._get_coef_matrix(features, cr, dr)

    # Transform the conditionally sorted indexing matrix.
    indexing = self._transform_indexing_matrix(indexing)

    # Convert the data types
    features = features.astype(np.float32)
    if self._atomic_forces:
      coef = coef.astype(np.float32)
      indexing = indexing.astype(np.int32)

    return features, coef, indexing


class MultiTransformer:
  """
  A flexible transformer targeting on AxByCz ... molecular compositions.
  """

  def __init__(self, atom_types, k_max=3, max_occurs=None, norm='exp',
               norm_order=1, include_all_k=True, periodic=False, lj=False,
               atomic_forces=False, cutoff=None):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the atomic species.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      max_occurs: a `Dict[str, int]` as the maximum appearances for a specie. 
        If an atom is explicitly specied, it can appear infinity times.
      norm: a `str` specifying the normalization function to use. Defaults to
        `exp`, `lj`, `exp+g` and `morse` are also supported.
      norm_order: an `int` or a `float` as the order for normalizing interatomic
        distances.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
      periodic: a `bool` indicating whether we shall use periodic boundary 
        conditions.
      atomic_forces: a `bool` indicating whether the atomic forces derivation is
        enabled or not.
      lj: a `bool` indicating that this transformer targets on LJ systems.
      cutoff: a `float` as the cutoff.

    """
    # Make sure the ghost atom is always the last one!
    if include_all_k and k_max == 3:
      num_ghosts = 1
      atom_types = list(atom_types)
      if GHOST in atom_types:
        if atom_types[-1] != GHOST:
          atom_types.remove(GHOST)
          atom_types = sorted(atom_types) + [GHOST]
        else:
          atom_types = sorted(atom_types[:-1]) + [GHOST]
      else:
        atom_types = sorted(atom_types) + [GHOST]
    elif k_max > 3:
      raise ValueError("k_max > 3 is not supported!")
    else:
      num_ghosts = 0
      if GHOST in atom_types:
        raise ValueError("GHOST is not allowed when k_max == 2!")

    # Determine the species and maximum occurs.
    species = []
    max_occurs = {} if max_occurs is None else dict(max_occurs)
    max_occurs[GHOST] = num_ghosts
    for specie in atom_types:
      species.extend(list(repeat(specie, max_occurs.get(specie, k_max))))
      if specie not in max_occurs:
        max_occurs[specie] = np.inf

    self._include_all_k = include_all_k
    self._k_max = k_max
    self._atom_types = atom_types
    self._species = species
    self._num_atom_types = len(atom_types)
    self._num_ghosts = num_ghosts
    self._kbody_terms = get_kbody_terms_from_species(species, k_max)
    self._transformers = {}
    self._max_occurs = max_occurs
    self._norm_order = norm_order
    self._periodic = periodic
    self._atomic_forces = atomic_forces
    self._lj = lj
    self._norm = norm
    self._cutoff = cutoff

    # The global split dims is None so that internal `_Transformer` objects will
    # construct their own `splid_dims`.
    self._split_dims = None

  @property
  def k_max(self):
    """
    Return the many-body expansion factor.
    """
    return self._k_max

  @property
  def included_k(self):
    """
    Return the included k.
    """
    if self._include_all_k:
      return list(range(1, self._k_max + 1))
    else:
      return [1, self._k_max]

  @property
  def ck2(self):
    """
    Return the value of C(k,2).
    """
    return comb(self._k_max, 2, exact=True)

  @property
  def kbody_terms(self):
    """
    Return the ordered k-body terms for this transformer.
    """
    return self._kbody_terms

  @property
  def species(self):
    """
    Return the ordered species of this transformer.
    """
    return self._species

  @property
  def atom_types(self):
    """
    Return the supported atom types.
    """
    return self._atom_types

  @property
  def number_of_atom_types(self):
    """
    Return the number of atom types in this transformer.
    """
    return self._num_atom_types

  @property
  def include_all_k(self):
    """
    Return True if a standalone two-body term is included.
    """
    return self._include_all_k

  @property
  def is_periodic(self):
    """
    Return True if this is a periodic transformer.
    """
    return self._periodic

  @property
  def max_occurs(self):
    """
    Return the maximum occurances of each type of atom.
    """
    return self._max_occurs

  @property
  def atomic_forces_enabled(self):
    """
    Return True if atomic forces are also computed.
    """
    return self._atomic_forces

  @property
  def is_lj(self):
    """
    Return True if this transformer is used for LJ systems.
    """
    return self._lj

  @property
  def cutoff(self):
    """
    Return the cutoff.
    """
    return self._cutoff

  def accept_species(self, species):
    """
    Return True if the given species can be handled.
    
    Args:
      species: a `List[str]` as the ordered species of a molecule.
    
    Returns:
      accepted: True if the given species can be handled by this transformer.
    
    """
    counter = Counter(species)
    return all(counter[e] <= self._max_occurs.get(e, 0) for e in counter)

  def _get_transformer(self, species):
    """
    Return the `Transformer` for the given list of species.
    
    Args:
      species: a `List[str]` as the atomic species.

    Returns:
      clf: a `Transformer`.

    """
    species = list(species) + [GHOST] * self._num_ghosts
    formula = get_formula(species)
    clf = self._transformers.get(
      formula, Transformer(species=species,
                           k_max=self._k_max,
                           kbody_terms=self._kbody_terms,
                           split_dims=self._split_dims,
                           norm=self._norm,
                           norm_order=self._norm_order,
                           periodic=self._periodic,
                           atomic_forces=self._atomic_forces,
                           lj=self._lj,
                           cutoff=self._cutoff)
    )
    self._transformers[formula] = clf
    return clf

  def transform_trajectory(self, trajectory):
    """
    Transform the given trajectory (a list of `ase.Atoms` with the same chemical
    symbols or an `ase.io.TrajectoryReader`) to input features.

    Args:
      trajectory: a `list` of `ase.Atoms` or a `ase.io.TrajectoryReader`. All
        objects should have the same chemical symbols.

    Returns:
      sample: a `KcnnSample` object.

    """
    ntotal = len(trajectory)
    assert ntotal > 0

    species = trajectory[0].get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    nrows, ncols = clf.shape
    features = np.zeros((ntotal, nrows, ncols), dtype=np.float32)

    if self._atomic_forces:
      coef = np.zeros_like((ntotal, nrows, ncols * 6), dtype=np.float32)
      num_force_components = 3 * len(species)
      num_entries = nrows * ncols * 6 // num_force_components
      indexing = np.zeros((ntotal, num_force_components, num_entries),
                          dtype=np.float32)
    else:
      coef = None
      indexing = None

    occurs = np.zeros((ntotal, self._num_atom_types), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._atom_types.index(specie)
      if loc < 0 or loc >= self._num_atom_types:
        raise ValueError("The loc of {:s} is {:d}!".format(specie, loc))
      occurs[:, loc] = float(times)

    weights = np.zeros((ntotal, nrows), dtype=np.float32)
    compress_stats = {}

    for i, atoms in enumerate(trajectory):
      _, coef_, indexing_ = clf.transform(atoms, features=features[i])
      if self._atomic_forces:
        # TODO: fix the bug
        coef[i] = coef_
        indexing[i] = indexing_
      if self._cutoff is None:
        weights[i] = clf.binary_weights
      else:
        weights[i], stats = clf.compress(features[i])
        for k, v in stats.items():
          compress_stats[k] = max(compress_stats.get(k, 0), v)

    return KcnnSample(features=features,
                      split_dims=split_dims,
                      binary_weights=weights,
                      occurs=occurs,
                      coefficients=coef,
                      indexing=indexing,
                      compress_stats=compress_stats)

  def transform(self, atoms):
    """
    Transform a single `ase.Atoms` object to input features.

    Args:
      atoms: an `ase.Atoms` object as the target to transform.

    Returns:
      sample: a `KcnnSample` object.
    
    Raises:
      ValueError: if the `species` is not supported by this transformer.

    """
    species = atoms.get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    features, coef, indexing = clf.transform(atoms)
    occurs = np.zeros((1, self._num_atom_types), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._atom_types.index(specie)
      if loc < 0:
        raise ValueError("The loc of %s is -1!" % specie)
      occurs[0, loc] = float(times)
    if self._cutoff is not None:
      weights, compress_stats = clf.compress(features)
    else:
      weights = np.array(clf.binary_weights)
      compress_stats = {}
    return KcnnSample(features=features,
                      split_dims=split_dims,
                      binary_weights=weights,
                      occurs=occurs,
                      coefficients=coef,
                      indexing=indexing,
                      compress_stats=compress_stats)

  def compute_atomic_energies(self, species, y_kbody, y_atomic_1body):
    """
    Compute the atomic energies given predicted kbody contributions.
    
    Args:
      species: a `List[str]` as the ordered atomic species.
      y_kbody: a `float32` array of shape `[num_examples, N]` as the k-body
        contribs.
      y_atomic_1body: a `Dict[str, float]` as the one-body energy of each kind 
        of atom.

    Returns:
      y_atomic: a `float32` array of shape `[num_examples, num_atoms]`.

    """

    # Get the feature transformer
    clf = self._get_transformer(species)

    # Allocate arrays
    num_examples = y_kbody.shape[0]
    num_atoms = len(species)
    split_dims = clf.split_dims
    y_atomic = np.zeros((num_examples, num_atoms))

    # Setup the 1-body atomic energies.
    for i in range(num_atoms):
      y_atomic[:, i] = y_atomic_1body[species[i]]

    # Compute and add the higher order (k = 2, 3, ...) corrections.
    for step in range(num_examples):
      for i, kbody_term in enumerate(clf.kbody_terms):
        if kbody_term not in clf.kbody_selections:
          continue

        # Compute the real `k` for this k-body term by excluding ghost atoms.
        symbols = get_atoms_from_kbody_term(kbody_term)
        k = len(symbols) - symbols.count(GHOST)

        # For each k-atoms selection, its energy should contrib equally to the
        # selected atoms. In my paper the coef should be `1 / factorial(k)` but
        # since our k-atom selections are all unique so coef here is `1 / k`.
        coef = 1.0 / k

        # Locate the starting index
        istart = 0 if i == 0 else int(sum(split_dims[:i]))

        # Loop through all k-atoms selections
        for ki, indices in enumerate(clf.kbody_selections[kbody_term]):
          for ai in indices[:k]:
            y_atomic[step, ai] += y_kbody[step, istart + ki] * coef
    return y_atomic


class OneBodyCalculator:
  """
  A helper class to compute the initial one-body weights.
  """

  def __init__(self, atom_types, num_examples, algorithm='default', factor=1.0,
               include_perturbations=True):
    """
    Initialization method.

    Args:
      atom_types: a list of `str` as the types of atoms.
      num_examples: an `int` as the total number of examples.
      algorithm: a `str` as the algorithm to compute the one-body weights.
      factor: a `float` as the scaling factor as the one-body weights.
      include_perturbations: a `bool`. If True, the higher-order perturbations
        terms will be included in the coefficients matrix as well.

    """
    self.atom_types = atom_types
    self.num_atom_types = len(atom_types)
    if atom_types[-1] == GHOST:
      self.num_real_atom_types = self.num_atom_types - 1
    else:
      self.num_real_atom_types = self.num_atom_types
    self.minima = {}
    self.b = np.zeros((num_examples, ))
    self.algorithm = algorithm.lower()
    self.factor = factor
    self.include_perturbations = include_perturbations
    self.mp2 = self.num_real_atom_types
    self.mp3 = self.num_real_atom_types + 1
    if not include_perturbations:
      self.coef = np.zeros((num_examples, self.num_real_atom_types))
    else:
      self.coef = np.zeros((num_examples, self.num_real_atom_types + 2))

  def add(self, index, chemical_symbols, y_true):
    """
    Add an example.

    Args:
      index: an `int` as the index of this sample.
      chemical_symbols: a `list` of `str` as the chemical symbols of this
        example.
      y_true: a `float` as the total energy of this example.

    """
    counter = Counter(chemical_symbols)
    for loc, atom in enumerate(self.atom_types[:self.num_real_atom_types]):
      self.coef[index, loc] = counter[atom]
    if self.include_perturbations:
      self.coef[index, self.mp2] = comb(len(chemical_symbols), 2)
      self.coef[index, self.mp3] = comb(len(chemical_symbols), 3)
    self.b[index] = y_true
    sch = self.get_stoichiometry(self.coef[index, :self.num_real_atom_types])
    if sch not in self.minima or y_true < self.b[self.minima[sch]]:
      self.minima[sch] = index

  def compute(self):
    """
    Compute the one-body weights.
    """
    if self.algorithm == 'minimal':
      # Only select the values from the global minima.
      selected = np.ix_(list(self.minima.values()))
      coef = self.coef[selected]
      b = self.b[selected]
    else:
      coef = self.coef
      b = self.b
    # The size of `x` is always equal to `self.num_real_atom_types`. We may need
    # to pad an zero at the end.
    x = _compute_lr_weights(
      coef, b,
      num_real_atom_types=self.num_real_atom_types,
      factor=self.factor
    )
    x = np.resize(x, self.num_atom_types)
    x[self.num_real_atom_types:] = 0.0
    return x

  def get_stoichiometry(self, atoms_counts):
    """
    A helper function to get the stoichiometry of a structure.
    """
    return ";".join(["{},{}".format(self.atom_types[j], int(atoms_counts[j]))
                     for j in range(self.num_real_atom_types)])


class FixedLenMultiTransformer(MultiTransformer):
  """
  This is also a flexible transformer targeting on AxByCz ... molecular 
  compositions but the length of the transformed features are the same so that
  they can be used for training.
  """

  def __init__(self, max_occurs, periodic=False, k_max=3, norm='exp',
               norm_order=1, include_all_k=True, atomic_forces=False, lj=False,
               cutoff=None):
    """
    Initialization method. 
    
    Args:
      max_occurs: a `Dict[str, int]` as the maximum appearances for each kind of 
        atomic specie.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      norm: a `str` specifying the normalization function to use. Defaults to
        `exp`, `lj`, `exp+g` and `morse` are also supported.
      norm_order: an `int` or a `float` as the order for normalizing interatomic
        distances.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
      atomic_forces: a `bool` indicating whether the atomic forces derivation is
        enabled or not.
      lj: a `bool` indicating that this transformer targets on LJ systems.
      cutoff: a `float` as the cutoff.
    
    """
    super(FixedLenMultiTransformer, self).__init__(
      atom_types=list(max_occurs.keys()),
      k_max=k_max,
      max_occurs=max_occurs,
      norm=norm,
      norm_order=norm_order,
      include_all_k=include_all_k,
      periodic=periodic,
      atomic_forces=atomic_forces,
      lj=lj,
      cutoff=cutoff
    )
    self._split_dims = self._get_fixed_split_dims()
    self._total_dim = sum(self._split_dims)

  @property
  def shape(self):
    """
    Return the shape of input feature matrix of this transformer.
    """
    return self._total_dim, comb(self._k_max, 2, exact=True)

  @property
  def split_dims(self):
    """
    Return the fixed `split_dims` for all internal transformers.
    """
    return self._split_dims

  def _get_fixed_split_dims(self):
    """
    The `split_dims` of all `_Transformer` should be the same.
    """
    split_dims = []
    for kbody_term in self._kbody_terms:
      atoms = get_atoms_from_kbody_term(kbody_term)
      counter = Counter(atoms)
      dims = [comb(self._max_occurs[e], k, True) for e, k in counter.items()]
      split_dims.append(np.prod(dims))
    return [int(x) for x in split_dims]

  def _log_compression_results(self, results):
    """
    A helper function to log the compression result.
    """
    print("The soft compression algorithm is applied with cutoff = "
          "{:.2f}".format(self._cutoff))

    num_loss_total = 0
    num_total = self._total_dim
    for j, kbody_term in enumerate(self._kbody_terms):
      num_full = self._split_dims[j]
      num_kept = results.get(kbody_term, 0)
      num_loss = num_full - num_kept
      num_loss_total += num_loss
      print("{:<12s} : {:5d} / {:5d}, compression = {:.2f}%".format(
        kbody_term, num_loss, num_full, num_loss / num_full * 100))
    print("Final result : {:5d} / {:5d}, compression = {:.2f}%".format(
      num_loss_total, num_total, num_loss_total / num_total * 100))

  def _transform_and_save(self, filename, examples, num_examples, max_size,
                          loss_fn=None, verbose=True, one_body_kwargs=None):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      filename: a `str` as the file to save examples.
      examples: a iterator which iterates through all examples.
      num_examples: an `int` as the number of examples.
      max_size: an `int` as the maximum size of all structures. This determines
        the dimension of the forces.
      verbose: boolean indicating whether.
      loss_fn: a `Callable` for transforming the calculated raw loss.
      one_body_kwargs: a `dict` as the key-value args for computing initial
        one-body weights.

    Returns:
      weights: a `float32` array as the weights for linear fit of the energies.

    """

    def _identity(_):
      """
      An identity function which returns the input directly.
      """
      return 1.0

    # Setup the loss function.
    loss_fn = loss_fn or _identity

    # Setup the one-body weights calculator
    one_body = OneBodyCalculator(
      self._atom_types, num_examples, **(one_body_kwargs or {}))

    # Start the timer
    tic = time.time()

    with tf.python_io.TFRecordWriter(filename) as writer:
      if verbose:
        print("Start transforming {} ... ".format(filename))

      compress_stats = {}

      for i, atoms in enumerate(examples):

        species = atoms.get_chemical_symbols()
        y_true = atoms.get_total_energy()
        sample = self.transform(atoms)

        x = _bytes_feature(sample.features.tostring())
        y = _bytes_feature(np.atleast_2d(-y_true).tostring())
        z = _bytes_feature(sample.occurs.tostring())
        w = _bytes_feature(sample.binary_weights.tostring())
        y_weight = _float_feature(loss_fn(y_true))

        if not self._atomic_forces:
          example = Example(
            features=Features(feature={'energy': y, 'features': x, 'occurs': z,
                                       'weights': w, 'loss_weight': y_weight}))
        else:
          # Pad zeros to the forces so that all forces of this dataset have the
          # same dimension.
          forces = atoms.get_forces()
          pad = max_size - len(forces)
          if pad > 0:
            forces = np.pad(forces, ((0, pad), (0, 0)), mode='constant')
          forces = _bytes_feature(forces.flatten().tostring())
          coef = _bytes_feature(sample.coefficients.tostring())
          indexing = _bytes_feature(sample.indexing.tostring())
          example = Example(
            features=Features(feature={'energy': y, 'features': x, 'occurs': z,
                                       'weights': w, 'loss_weight': y_weight,
                                       'indexing': indexing, 'coef': coef,
                                       'forces': forces}))
        writer.write(example.SerializeToString())

        # Add this example to the one-body database
        one_body.add(i, species, y_true)

        # Save the compress stats for this example
        for k, v in sample.compress_stats.items():
          compress_stats[k] = max(compress_stats.get(k, 0), v)

        if verbose and (i + 1) % 100 == 0:
          sys.stdout.write("\rProgress: {:7d} / {:7d} | Speed = {:6.1f}".format(
            i + 1, num_examples, (i + 1) / (time.time() - tic)))

      if verbose:
        print("")
        print("Transforming {} finished!".format(filename))

        if self._cutoff is not None:
          self._log_compression_results(compress_stats)

      return one_body.compute()

  def _save_auxiliary_for_file(self, filename, max_size, lookup_indices=None,
                               initial_1body_weights=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      max_size: an `int` as maximum size of all structures. This determines the
        dimension of the forces.
      initial_1body_weights: a `float32` array of shape `[num_atom_types, ]` as
        the initial weights for the one-body convolution kernels.
      lookup_indices: a `List[int]` as the indices of each given example.

    """
    if lookup_indices is not None:
      lookup_indices = list(lookup_indices)
    else:
      lookup_indices = []

    if initial_1body_weights is not None:
      initial_1body_weights = list(initial_1body_weights)
    else:
      initial_1body_weights = []

    max_occurs = {atom: times for atom, times in self._max_occurs.items()
                  if times < self._k_max}
    num_entries = _get_num_force_entries(max_size, self._k_max)

    auxiliary_properties = {
      "kbody_terms": self._kbody_terms,
      "split_dims": self._split_dims,
      "shape": self.shape,
      "lookup_indices": list([int(i) for i in lookup_indices]),
      "atom_types": self._atom_types,
      "num_atom_types": self._num_atom_types,
      "species": self._species,
      "include_all_k": self._include_all_k,
      "periodic": self._periodic,
      "k_max": self._k_max,
      "max_occurs": max_occurs,
      "norm": self._norm,
      "norm_order": self._norm_order,
      "initial_one_body_weights": initial_1body_weights,
      "atomic_forces_enabled": self._atomic_forces,
      "indexing_shape": [max_size * 3, num_entries],
      "lj": self._lj,
      "cutoff": self._cutoff
    }

    with open(join(dirname(filename),
                   "{}.json".format(splitext(basename(filename))[0])),
              "w+") as fp:
      json.dump(auxiliary_properties, fp=fp, indent=2)

  def transform_and_save(self, database, train_file=None, test_file=None,
                         loss_fn=None, verbose=True, one_body_kwargs=None):
    """
    Transform coordinates to input features and save them to tfrecord files
    using `tf.TFRecordWriter`.

    Args:
      database: a `Database` as the parsed results from a xyz file.
      train_file: a `str` as the file for saving training data or None to skip.
      test_file: a `str` as the file for saving testing data or None to skip.
      verbose: a `bool` indicating whether logging the transformation progress.
      loss_fn: a `Callable` for computing the exponential scaled RMSE loss.
      one_body_kwargs: a `dict` as the configs for the initial one-body weigts
        calculator.

    """
    sizes = database.get_atoms_size_distribution()
    max_size = max(sizes)

    if test_file:
      examples = database.examples(mode=tf.estimator.ModeKeys.EVAL)
      id_list = database.ids_of_testing_examples
      num_examples = len(id_list)
      if num_examples > 0:
        self._transform_and_save(
          test_file,
          examples,
          num_examples,
          max_size,
          loss_fn=loss_fn,
          verbose=verbose
        )
        self._save_auxiliary_for_file(
          test_file,
          max_size=max_size,
          lookup_indices=id_list
        )

    if train_file:
      examples = database.examples(mode=tf.estimator.ModeKeys.TRAIN)
      id_list = database.ids_of_training_examples
      num_examples = len(id_list)
      if num_examples > 0:
        weights = self._transform_and_save(
          train_file,
          examples,
          num_examples,
          max_size,
          loss_fn=loss_fn,
          verbose=verbose,
          one_body_kwargs=one_body_kwargs or {}
        )
        self._save_auxiliary_for_file(
          train_file,
          max_size=max_size,
          initial_1body_weights=weights,
          lookup_indices=id_list
        )
