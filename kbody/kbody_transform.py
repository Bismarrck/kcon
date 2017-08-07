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
from collections import Counter
from itertools import combinations, product, repeat, chain
from os.path import basename, dirname, join, splitext
from ase.atoms import Atoms
from scipy.misc import comb
from sklearn.metrics import pairwise_distances
from tensorflow.python.training.training import Features, Example
from constants import pyykko, GHOST
from utils import get_atoms_from_kbody_term

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS


def get_formula(species):
  """
  Return the molecular formula given a list of atomic species.
  """
  return "".join(species)


def _compute_lr_weights(coef, y):
  """
  Solve the linear equation system of Ax = b.
  
  Args:
    coef: a `float` array of shape `[num_examples, num_atom_types]`.
    y: a `float` array of shape `[num_examples, ]`.

  Returns:
    x: a `float` array of shape `[num_atom_types, ]` as the solution.

  """
  return np.negative(np.dot(np.linalg.pinv(coef), y))


def _get_pyykko_bonds_matrix(species, factor=1.0, flatten=True):
  """
  Return the pyykko-bonds matrix given a list of atomic symbols.

  Args:
    species: a `List[str]` as the atomic symbols.
    factor: a `float` as the normalization factor.
    flatten: a `bool` indicating whether the bonds matrix is flatten or not.

  Returns:
    bonds: the bonds matrix (or vector if `flatten` is True).

  """
  rr = np.asarray([pyykko[specie] for specie in species])[:, np.newaxis]
  lmat = np.multiply(factor, rr + rr.T)
  if flatten:
    return lmat.flatten()
  else:
    return lmat


def _get_kbody_terms(species, many_body_k):
  """
  Return the k-body terms given the chemical symbols and `many_body_k`.

  Args:
    species: a `list` of `str` as the chemical symbols.
    many_body_k: a `int` as the maximum k-body terms that we should consider.

  Returns:
    kbody_terms: a `list` of `str` as the k-body terms.

  """
  return sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, many_body_k)])))


def exponential(x, l, order=1):
  """
  Normalize the `inputs` with the exponential function:
    f(r) = exp(-r/L)

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    l: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).
    order: a `int` as the exponential order. If `order` is 0, the inputs will 
      not be scaled by `factor`.

  Returns:
    scaled: the scaled inputs.

  """
  if order == 0:
    return np.exp(-x)
  else:
    return np.exp(-(x / l) ** order)


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


class _Transformer:
  """
  This class is used to transform atomic coordinates to input feature matrix.
  """

  def __init__(self, species, k_max=3, kbody_terms=None, split_dims=None,
               norm_order=1, periodic=False):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      kbody_terms: a `List[str]` as the k-body terms.
      split_dims: a `List[int]` as the dimensions for splitting inputs. If this
        is given, the `kbody_terms` must also be set and their lengths should be
        equal.
      norm_order: a `int` as the order for normalizing interatomic distances.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.

    """
    if split_dims is not None:
      assert len(split_dims) == len(kbody_terms)

    kbody_terms = kbody_terms or _get_kbody_terms(species, k_max)
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

    # Initialize internal variables.
    self._k_max = k_max
    self._kbody_terms = kbody_terms
    self._kbody_offsets = offsets
    self._kbody_sizes = kbody_sizes
    self._species = species
    self._mapping = mapping
    self._selections = selections
    self._split_dims = split_dims
    self._ck2 = int(comb(k_max, 2, exact=True))
    self._cond_sort = _get_conditional_sorting_indices(kbody_terms)
    self._normalizers = _get_pyykko_bonds_matrix(species)
    self._norm_order = norm_order
    self._num_ghosts = num_ghosts
    self._periodic = periodic
    self._binary_weights = self._get_binary_weights()
    self._real_dim = real_dim

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
      # get flatten lists, eg [1, 2, 8].
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
    offsets = self._kbody_offsets
    for i in range(len(self._split_dims)):
      weights[offsets[i]: offsets[i] + self._kbody_sizes[i]] = 1.0
    return weights

  def transform(self, atoms, out=None):
    """
    Transform the given `ase.Atoms` object to an input feature matrix.

    Args:
      atoms: an `ase.Atoms` object.
      out: a 2D `float32` array of None as the location into which the result is
        stored. If not provided, a new array will be allocated.

    Returns:
      out: a `float32` array of shape `self.shape` as the input feature matrix.

    """
    if out is None:
      out = np.zeros((self._real_dim, self._ck2), dtype=np.float32)
    elif out.shape != self.shape:
      raise ValueError("The shape should be {}".format(self.shape))

    if self._num_ghosts > 0:
      # Append `num_ghosts` rows of zeros to the positions. We can not directly
      # use `inf` because `pairwise_distances` and `get_all_distances` do not
      # support `inf`.
      aux_vecs = np.zeros((self._num_ghosts, 3))
      coords = np.vstack((atoms.get_positions(), aux_vecs))
    else:
      coords = atoms.get_positions()

    # Compute the interatomic distances. For non-periodic molecules we use the
    # faster method `pairwise_distances`.
    if not self.is_periodic:
      dists = pairwise_distances(coords)
    else:
      _atoms = Atoms(symbols=self._species,
                     positions=coords,
                     cell=atoms.get_cell(),
                     pbc=atoms.get_pbc())
      dists = _atoms.get_all_distances(mic=True)
      del _atoms

    # Manually set the distances between ghost atoms and real atoms to inf.
    if self._num_ghosts > 0:
      dists[:, -self._num_ghosts:] = np.inf
      dists[-self._num_ghosts:, :] = np.inf

    # Normalize the interatomic distances with the exponential function so that
    # shorter bonds have larger normalized weights.
    dists = dists.flatten()
    norm_dists = exponential(dists, self._normalizers, order=self._norm_order)

    # Assign the normalized distances to the input feature matrix.
    for i, kbody_term in enumerate(self._kbody_terms):
      if kbody_term not in self._mapping:
        continue
      # The index matrix was transposed because typically C(N, k) >> C(k, 2).
      # See `_get_mapping`.
      index_matrix = self._mapping[kbody_term]
      istart = self._offsets[i]
      # Manually adjust the step size because the offset length may be larger if
      # `split_dims` is fixed.
      istep = min(self._offsets[i + 1] - istart, index_matrix.shape[1])
      istop = istart + istep
      for k in range(self._ck2):
        out[istart: istop, k] = norm_dists[index_matrix[k]]

    # Conditional sorting.
    for i, kbody_term in enumerate(self._kbody_terms):
      if kbody_term not in self._mapping:
        continue
      for ix in self._cond_sort.get(kbody_term, []):
        # Note:
        # `samples` is a 2D array, the Python advanced slicing will make the
        # returned `z` a copy but not a view. The shape of `z` is transposed.
        # So we should sort along axis 0 here!
        z = out[self._offsets[i]: self._offsets[i + 1], ix]
        z.sort()
        out[self._offsets[i]: self._offsets[i + 1], ix].sort()

    return out


class MultiTransformer:
  """
  A flexible transformer targeting on AxByCz ... molecular compositions.
  """

  def __init__(self, atom_types, k_max=3, max_occurs=None, norm_order=1,
               include_all_k=True, periodic=False):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the atomic species.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      max_occurs: a `Dict[str, int]` as the maximum appearances for a specie. 
        If an atom is explicitly specied, it can appear infinity times.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
      periodic: a `bool` indicating whether we shall use periodic boundary 
        conditions.

    """

    # Determine the number of ghost atoms (k_max - 2) to add.
    if include_all_k and k_max >= 3:
      num_ghosts = k_max - 2
      if GHOST not in atom_types:
        atom_types = list(atom_types) + [GHOST] * num_ghosts
    else:
      num_ghosts = 0

    # Determine the species and maximum occurs.
    species = []
    max_occurs = {} if max_occurs is None else dict(max_occurs)
    if num_ghosts > 0:
      max_occurs[GHOST] = num_ghosts
    for specie in atom_types:
      species.extend(list(repeat(specie, max_occurs.get(specie, k_max))))
      if specie not in max_occurs:
        max_occurs[specie] = np.inf

    self._include_all_k = include_all_k
    self._k_max = k_max
    self._atom_types = list(set(atom_types))
    self._kbody_terms = _get_kbody_terms(species, k_max)
    self._transformers = {}
    self._max_occurs = max_occurs
    self._norm_order = norm_order
    self._species = sorted(list(max_occurs.keys()))
    self._num_atom_types = len(self._species)
    self._num_ghosts = num_ghosts
    self._periodic = periodic
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
      formula, _Transformer(species=species,
                            k_max=self._k_max,
                            kbody_terms=self._kbody_terms,
                            split_dims=self._split_dims,
                            norm_order=self._norm_order,
                            periodic=self._periodic)
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
      features: a `float32` 4D array as the input feature matrix.
      split_dims: a `List[int]` for splitting the input features along axis 2.
      weights: a 1D array as the binary weights of each row of the features.
      occurs: a 2D array as the occurances of the species.

    """
    ntotal = len(trajectory)
    assert ntotal > 0

    species = trajectory[0].get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    nrows, ncols = clf.shape
    features = np.zeros((ntotal, nrows, ncols), dtype=np.float32)
    occurs = np.zeros((ntotal, len(self._species)), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._species.index(specie)
      if loc < 0:
        raise ValueError("The loc of %s is -1!" % specie)
      occurs[:, loc] = float(times)
    for i, atoms in enumerate(trajectory):
      clf.transform(atoms, features[i])
    weights = np.tile(clf.binary_weights, (ntotal, 1))
    return features, split_dims, weights, occurs

  def transform(self, atoms):
    """
    Transform a single `ase.Atoms` object to input features.

    Args:
      atoms: an `ase.Atoms` object as the target to transform.

    Returns:
      features: a `float32` 4D array as the input feature matrix.
      split_dims: a `List[int]` for splitting the input features along axis 2.
      weights: a 1D array as the binary weights of each row of the features. 
      occurs: a 2D array as the occurances of the species.
    
    Raises:
      ValueError: if the `species` is not supported by this transformer.

    """
    species = atoms.get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    features = clf.transform(atoms)
    occurs = np.zeros((1, len(self._species)), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._species.index(specie)
      if loc < 0:
        raise ValueError("The loc of %s is -1!" % specie)
      occurs[0, loc] = float(times)
    weights = np.array(clf.binary_weights)
    return features, split_dims, weights, occurs

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
        k = len(kbody_term) - kbody_term.count(GHOST)
        # For each k-atoms selection, its energy should contrib equally to the
        # selected atoms.
        coef = 1.0 / k
        # Locate the starting index
        istart = 0 if i == 0 else int(sum(split_dims[:i]))
        # Loop through all k-atoms selections
        for ki, indices in enumerate(clf.kbody_selections[kbody_term]):
          for ai in indices[:k]:
            y_atomic[step, ai] += y_kbody[step, istart + ki] * coef
    return y_atomic


class FixedLenMultiTransformer(MultiTransformer):
  """
  This is also a flexible transformer targeting on AxByCz ... molecular 
  compositions but the length of the transformed features are the same so that
  they can be used for training.
  """

  def __init__(self, max_occurs, periodic=False, k_max=3, norm_order=1,
               include_all_k=False):
    """
    Initialization method. 
    
    Args:
      max_occurs: a `Dict[str, int]` as the maximum appearances for each kind of 
        atomic specie.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      norm_order: a `int` as the normalization order.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
    
    """
    super(FixedLenMultiTransformer, self).__init__(
      atom_types=list(max_occurs.keys()),
      k_max=k_max,
      max_occurs=max_occurs,
      norm_order=norm_order,
      include_all_k=include_all_k,
      periodic=periodic,
    )
    self._split_dims = self._get_fixed_split_dims()
    self._total_dim = sum(self._split_dims)

  @property
  def shape(self):
    """
    Return the shape of input feature matrix of this transformer.
    """
    return self._total_dim, comb(self._k_max, 2, exact=True)

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

  def _transform_and_save(self, filename, examples, num_examples, loss_fn=None,
                          verbose=True):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      filename: a `str` as the file to save examples.
      examples: a iterator which shall iterate through all samples.
      verbose: boolean indicating whether.
      loss_fn: a `Callable` for transforming the calculated raw loss.

    Returns:
      weights: a `float32` array as the weights for linear fit of the energies.

    """

    def _identity(v):
      """
      An identity function which returns the input directly.
      """
      return v

    # Setup the loss function.
    loss_fn = loss_fn or _identity

    with tf.python_io.TFRecordWriter(filename) as writer:
      if verbose:
        print("Start mixed transforming %s ... " % filename)

      coef = np.zeros((num_examples, self.number_of_atom_types))
      b = np.zeros((num_examples, ))

      for i, atoms in enumerate(examples):

        species = atoms.get_chemical_symbols()
        y_true = atoms.get_total_energy()
        features, split_dims, binary_weights, occurs = self.transform(atoms)

        x = _bytes_feature(features.tostring())
        y = _bytes_feature(np.atleast_2d(-y_true).tostring())
        z = _bytes_feature(occurs.tostring())
        w = _bytes_feature(binary_weights.tostring())
        y_weight = _float_feature(loss_fn(y_true))
        example = Example(
          features=Features(feature={'energy': y, 'features': x, 'occurs': z,
                                     'weights': w, 'loss_weight': y_weight}))
        writer.write(example.SerializeToString())

        counter = Counter(species)
        for loc, atom in enumerate(self._species):
          coef[i, loc] = counter[atom]
        b[i] = y_true

        if verbose and i % 100 == 0:
          sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))

      if verbose:
        print("")
        print("Transforming %s finished!" % filename)

      return _compute_lr_weights(coef, b)

  def _save_auxiliary_for_file(self, filename, initial_1body_weights=None,
                               lookup_indices=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      initial_1body_weights: a 1D array of shape `[num_atom_types, ]` as the
        initial weights for the one-body convolution kernels.
      lookup_indices: a `List[int]` as the indices of each given example.

    """
    name = splitext(basename(filename))[0]
    filename = join(dirname(filename), "{}.json".format(name))

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

    aux_dict = {
      "kbody_terms": self._kbody_terms,
      "split_dims": self._split_dims,
      "shape": self.shape,
      "lookup_indices": list([int(i) for i in lookup_indices]),
      "num_atom_types": len(self._species),
      "species": self._species,
      "include_all_k": self._include_all_k,
      "periodic": self._periodic,
      "k_max": self._k_max,
      "max_occurs": max_occurs,
      "norm_order": self._norm_order,
      "initial_one_body_weights": initial_1body_weights,
    }

    with open(filename, "w+") as f:
      json.dump(aux_dict, fp=f, indent=2)

  def transform_and_save(self, database, train_file=None, test_file=None,
                         verbose=True, loss_fn=None):
    """
    Transform coordinates to input features and save them to tfrecord files
    using `tf.TFRecordWriter`.

    Args:
      database: a `Database` as the parsed results from a xyz file.
      train_file: a `str` as the file for saving training data or None to skip.
      test_file: a `str` as the file for saving testing data or None to skip.
      verbose: a `bool` indicating whether logging the transformation progress.
      loss_fn: a `Callable` for computing the exponential scaled RMSE loss.

    """
    if test_file:
      examples = database.examples(for_training=False)
      ids = database.ids_of_testing_examples
      num_examples = len(ids)
      self._transform_and_save(
        test_file, examples, num_examples, loss_fn=loss_fn, verbose=verbose)
      self._save_auxiliary_for_file(test_file, lookup_indices=ids)

    if train_file:
      examples = database.examples(for_training=True)
      ids = database.ids_of_training_examples
      num_examples = len(ids)
      weights = self._transform_and_save(
        train_file, examples, num_examples, loss_fn=loss_fn, verbose=verbose)
      self._save_auxiliary_for_file(
        train_file, initial_1body_weights=weights, lookup_indices=ids)
