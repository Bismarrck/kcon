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
    # If an atom appears more in the k-body term, we should discard this k-body
    # term, eg, there should be no CCC or CCH for `CH4`.
    if any([counter[e] > len(atom_index.get(e, [])) for e in atoms]):
      continue
    # ck2 is the number of bond types in this k-body term.
    ck2 = int(comb(len(atoms), 2, exact=True))
    # Sort the atoms
    sorted_atoms = sorted(counter.keys())
    # Construct the k-atoms selection candidates. For each type of atom we draw
    # N times where N is equal to `counter[atom]`. Thus, the candidate list can
    # be constructed:
    # [[[1, 2], [1, 3], [1, 4], ...], [[8], [9], [10], ...]]
    # The length of the candidates is equal to the number of atom types.
    k_atoms_candidates = [
      [list(o) for o in combinations(atom_index[e], counter[e])]
      for e in sorted_atoms
    ]
    # Construct the k-atoms selections. First, we get the `product` (See Python
    # official document for more info), eg [[1, 2], [8]]. Then `chain` it to get
    # flatten lists, eg [1, 2, 8].
    k_atoms_selections = [list(chain(*o)) for o in product(*k_atoms_candidates)]
    selections[kbody_term] = k_atoms_selections
    # cnk is the number of k-atoms selections.
    cnk = len(k_atoms_selections)
    # Construct the mapping from the interatomic distance matrix to the input
    # matrix. This procedure can greatly increase the transformation speed.
    # The basic idea is to fill the input feature matrix with broadcasting. The
    # N-by-N interatomic distance matrix is flatten to 1D vector. Then we can
    # fill the matrix like this:
    #   feature_matrix[:, col] = flatten_dist[[1,2,8,10,9,2,1,1]]
    mapping[kbody_term] = np.zeros((ck2, cnk), dtype=int)
    for i in range(cnk):
      for j, (vi, vj) in enumerate(combinations(k_atoms_selections[i], 2)):
        mapping[kbody_term][j, i] = vi * natoms + vj
  return mapping, selections


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
  This class is used to transform atomic coordinates and energies to input
  features and training targets.
  """

  def __init__(self, species, many_body_k=3, kbody_terms=None, split_dims=None,
               order=1, periodic=False):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      many_body_k: a `int` as the maximum order for the many-body expansion.
      kbody_terms: a `List[str]` as the k-body terms.
      split_dims: a `List[int]` as the dimensions for splitting inputs. If this
        is given, the `kbody_terms` must also be set and their lengths should be
        equal.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.

    """
    if split_dims is not None:
      assert len(split_dims) == len(kbody_terms)

    num_ghosts = list(species).count(GHOST)
    if num_ghosts != 0 and (num_ghosts > 2 or many_body_k - num_ghosts != 2):
      raise ValueError("The number of ghosts is wrong!")

    if kbody_terms is None:
      kbody_terms = sorted(list(set(
        [",".join(sorted(c)) for c in combinations(species, many_body_k)])))

    # Generate the mapping from the N-by-N interatomic distances matrix to the
    # [C(N,k), C(k,2)] input feature matrix and the indices selections for each
    # C(N,k) terms.
    mapping, selections = _get_mapping(species, kbody_terms)

    if split_dims is None:
      offsets, dim, kbody_sizes = [0], 0, []
      for term in kbody_terms:
        tsize = mapping[term].shape[1] if term in mapping else 0
        dim += max(tsize, 1)
        offsets.append(dim)
        kbody_sizes.append(tsize)
      split_dims = np.diff(offsets).tolist()
    else:
      offsets = [0] + np.cumsum(split_dims).tolist()
      dim = offsets[-1]
      kbody_sizes = []
      for term in kbody_terms:
        kbody_sizes.append(mapping[term].shape[1] if term in mapping else 0)

    multipliers = np.zeros(dim, dtype=np.float32)
    for i in range(len(split_dims)):
      multipliers[offsets[i]: offsets[i] + kbody_sizes[i]] = 1.0

    self._many_body_k = many_body_k
    self._kbody_terms = kbody_terms
    self._species = species
    self._dist2inputs_mapping = mapping
    self._selections = selections
    self._kbody_offsets = offsets
    self._split_dims = split_dims
    self._cnk = int(comb(len(species), many_body_k, exact=True))
    self._ck2 = int(comb(many_body_k, 2, exact=True))
    self._sorting_indices = _get_conditional_sorting_indices(kbody_terms)
    self._lmat = _get_pyykko_bonds_matrix(species)
    self._kbody_sizes = kbody_sizes
    self._multipliers = multipliers
    self._order = order
    self._num_ghosts = num_ghosts
    self._periodic = periodic

    # The major dimension of each input feature matrix. Each missing kbody term
    # will be assigned with a zero row vector. `len(kbody_terms) - len(mapping)`
    # calculated the number of missing kbody terms.
    self._total_dim = dim

  @property
  def cnk(self):
    """
    Return the value of C(N,k) for this transformer.
    """
    return self._cnk

  @property
  def ck2(self):
    """
    Return the value of C(k,2) for this transformer.
    """
    return self._ck2

  @property
  def many_body_k(self):
    """
    Return the maximum order for the many-body expansion.
    """
    return self._many_body_k

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
  def multipliers(self):
    """
    Return the multipliers for the all k-body contribs.
    """
    return self._multipliers

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

  def transform(self, array_of_coords, y_true=None, array_of_lattice=None,
                array_of_pbc=None):
    """
    Transform the given atomic coordinates and energies to input features and
    training targets and return them as numpy arrays.

    Args:
      array_of_coords: a `float32` array of shape `[-1, N, 3]` as the atomic
        coordinates of sturctures.
      y_true: a `float64` array as the true energies.
      array_of_lattice: a `float32` array of shape `[-1, 9]` as the 3x3 lattice
        cells for all structures.
      array_of_pbc: a `bool` array of shape `[-1, 3]` as the periodic conditions
        along X,Y,Z directions for all structures.

    Returns:
      features: a 4D array as the transformed input features.
      targets: a 1D array as the training targets (actually the negative of the
        input energies) given `energies` or None.

    """
    num_examples = len(array_of_coords)
    samples = np.zeros((num_examples, self._total_dim, self.ck2),
                       dtype=np.float32)
    kbody_terms = self._kbody_terms
    mapping = self._dist2inputs_mapping
    offsets = self._kbody_offsets

    num_ghosts = self._num_ghosts
    if num_ghosts > 0:
      gvecs = np.zeros((self._num_ghosts, 3))
    else:
      gvecs = None

    for i in range(num_examples):
      if not self.is_periodic:
        if num_ghosts > 0:
          coords = np.vstack((array_of_coords[i], gvecs))
          dists = pairwise_distances(coords)
        else:
          dists = pairwise_distances(array_of_coords[i])
      # For periodic structures, the minimum image convention is needed.
      else:
        if num_ghosts > 0:
          coords = np.vstack((array_of_coords[i], gvecs))
        else:
          coords = array_of_coords[i]
        cell = array_of_lattice[i].reshape((3, 3))
        atoms = Atoms(self._species, coords, cell=cell, pbc=array_of_pbc[i])
        dists = atoms.get_all_distances(mic=True)

      # Manually set the distances of real atoms and the ghosts to infinity.
      if num_ghosts > 0:
        dists[:, -num_ghosts:] = np.inf
        dists[-num_ghosts:, :] = np.inf

      # Normalize the distances.
      dists = dists.flatten()
      rr = exponential(dists, self._lmat, order=self._order)

      samples[i].fill(0.0)
      for j, term in enumerate(kbody_terms):
        if term not in mapping:
          continue
        # Manually adjust the step size because when `split_dims` is fixed the
        # offset length may be bigger.
        istart = offsets[j]
        istep = min(offsets[j + 1] - istart, mapping[term].shape[1])
        istop = istart + istep
        for k in range(self.ck2):
          samples[i, istart: istop, k] = rr[mapping[term][k]]

      for j, term in enumerate(kbody_terms):
        if term not in mapping:
          continue
        for ix in self._sorting_indices.get(term, []):
          # Note:
          # `samples` is a 3D array, the Python advanced slicing will make the
          # returned `z` a copy but not a view. The shape of `z` is transposed.
          # So we should sort along axis 0 here!
          z = samples[i, offsets[j]: offsets[j + 1], ix]
          z.sort(axis=0)
          samples[i, offsets[j]: offsets[j + 1], ix] = z

    if y_true is not None:
      targets = np.negative(y_true)
    else:
      targets = None
    return samples, targets


class MultiTransformer:
  """
  A flexible transformer targeting on AxByCz ... molecular compositions.
  """

  def __init__(self, atom_types, many_body_k=3, max_occurs=None, order=1,
               two_body=False, periodic=False):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the target atomic species.  
      many_body_k: a `int` as the many body expansion factor.
      max_occurs: a `Dict[str, int]` as the maximum appearances for a specie. 
        If an atom is explicitly specied, it can appear infinity times.
      two_body: a `bool` indicating whether we shall use a standalone two-body
        term or not..
      periodic: a `bool` indicating whether we shall use periodic boundary 
        conditions.

    """
    if two_body and many_body_k >= 3:
      num_ghosts = many_body_k - 2
      if GHOST not in atom_types:
        atom_types = list(atom_types) + [GHOST] * num_ghosts
    else:
      num_ghosts = 0
    self._many_body_k = many_body_k
    self._atom_types = list(set(atom_types))
    species = []
    max_occurs = {} if max_occurs is None else dict(max_occurs)
    if num_ghosts > 0:
      max_occurs[GHOST] = num_ghosts
    for specie in self._atom_types:
      species.extend(list(repeat(specie, max_occurs.get(specie, many_body_k))))
      if specie not in max_occurs:
        max_occurs[specie] = np.inf
    self._kbody_terms = sorted(list(
        set([",".join(sorted(c)) for c in combinations(species, many_body_k)])))
    self._transformers = {}
    self._max_occurs = max_occurs
    self._order = order
    self._ordered_species = sorted(list(max_occurs.keys()))
    self._nat = len(self._ordered_species)
    self._num_ghosts = num_ghosts
    self._periodic = periodic

  @property
  def many_body_k(self):
    """
    Return the many-body expansion factor.
    """
    return self._many_body_k

  @property
  def ck2(self):
    """
    Return the value of C(k,2).
    """
    return int(comb(self._many_body_k, 2, exact=True))

  @property
  def kbody_terms(self):
    """
    Return the ordered k-body terms for this transformer.
    """
    return self._kbody_terms

  @property
  def ordered_species(self):
    """
    Return the ordered species of this transformer.
    """
    return self._ordered_species

  @property
  def number_of_atom_types(self):
    """
    Return the number of atom types in this transformer.
    """
    return self._nat

  @property
  def two_body(self):
    """
    Return True if a standalone two-body term is included.
    """
    return self._num_ghosts > 0

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
    Return True if the given species can be transformed by this transformer.
    
    Args:
      species: a `List[str]` as the ordered species of a molecule.
    
    Returns:
      accepted: True if this transformer can handle this species list.
    
    """
    counter = Counter(species)
    return all(counter[e] <= self._max_occurs.get(e, 0) for e in counter)

  def _get_transformer(self, species):
    """
    Return the `Transformer` for the given molecular species.
    
    Args:
      species: a `List[str]` as the ordered atomic species for molecules. 

    Returns:
      clf: a `Transformer`.

    """

    if self._num_ghosts > 0:
      species = list(species) + [GHOST] * self._num_ghosts

    formula = get_formula(species)
    clf = self._transformers.get(formula)
    if not isinstance(clf, _Transformer):
      clf = _Transformer(
        species,
        self.many_body_k,
        kbody_terms=self.kbody_terms,
        order=self._order,
        periodic=self._periodic
      )
      self._transformers[formula] = clf
    return clf

  def transform(self, species, array_of_coords, energies=None,
                array_of_lattice=None, array_of_pbc=None):
    """
    Transform the atomic coordinates to input features. All input structures
    must have the same atomic species.

    Args:
      species: a `List[str]` as the ordered atomic species for all `coords`.
      array_of_coords: a `float32` array of shape `[-1, num_atoms, 3]` as the
        atomic coordinates.
      energies: a `float64` array of shape `[-1, ]` as the true energies.
      array_of_lattice: a `float32` array of shape `[-1, 9]` as the periodic
        cell parameters for each structure.
      array_of_pbc: a `bool` array of shape `[-1, 3]` as the periodic conditions
        along XYZ directions.

    Returns:
      features: a 4D array as the input features.
      split_dims: a `List[int]` for splitting the input features along axis 2.
      targets: a 1D array as the training targets given `energis` or None.
      weights: a 1D array as the binary weights of each row of the features. 
      occurs: a 2D array as the occurances of the species.
    
    Raises:
      ValueError: if the `species` is not supported by this transformer.

    """

    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    array_of_coords = np.asarray(array_of_coords)
    if len(array_of_coords.shape) == 2:
      if array_of_coords.shape[0] != len(species):
        raise ValueError("The shapes of coords and species are not matched!")
      array_of_coords = array_of_coords.reshape((1, len(species), 3))
    elif array_of_coords.shape[1] != len(species):
      raise ValueError("The shapes of coords and species are not matched!")

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    features, targets = clf.transform(
      array_of_coords,
      energies,
      array_of_lattice=array_of_lattice,
      array_of_pbc=array_of_pbc
    )

    ntotal = array_of_coords.shape[0]
    occurs = np.zeros((ntotal, len(self._ordered_species)), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._ordered_species.index(specie)
      if loc < 0:
        raise ValueError("The loc of %s is -1!" % specie)
      occurs[:, loc] = float(times)

    weights = np.tile(clf.multipliers, (ntotal, 1))
    return features, split_dims, targets, weights, occurs

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

  def __init__(self, max_occurs, periodic=False, many_body_k=3, order=1,
               two_body=False):
    """
    Initialization method. 
    
    Args:
      max_occurs: a `Dict[str, int]` as the maximum appearances for each kind of 
        atomic specie.
      periodic: a `bool` indicating whether this transformer is used for 
        periodic structures or not.
      many_body_k: a `int` as the many body expansion factor.
      order: a `int` as the feature exponential order. 
      two_body: a `bool` indicating whether we shall use a standalone two-body
        term or not..
    
    """
    super(FixedLenMultiTransformer, self).__init__(
      list(max_occurs.keys()),
      many_body_k=many_body_k,
      max_occurs=max_occurs,
      order=order,
      two_body=two_body,
      periodic=periodic,
    )
    self._split_dims = self._get_fixed_split_dims()
    self._total_dim = sum(self._split_dims)

  @property
  def total_dim(self):
    """
    Return the total dimension of the transformed features.
    """
    return self._total_dim

  def _get_fixed_split_dims(self):
    """
    The `split_dims` for all `Transformer`s of this is fixed. 
    """
    split_dims = []
    for term in self._kbody_terms:
      counter = Counter(term.split(","))
      dims = [comb(self._max_occurs[e], k, True) for e, k in counter.items()]
      split_dims.append(np.prod(dims))
    return [int(x) for x in split_dims]

  def _get_transformer(self, species):
    """
    Return the `Transformer` for the given molecular species.

    Args:
      species: a `List[str]` as the ordered atomic species for molecules. 

    Returns:
      clf: a `Transformer`.

    """

    if self._num_ghosts > 0:
      species = list(species) + [GHOST] * self._num_ghosts

    formula = get_formula(species)
    clf = self._transformers.get(formula)
    if not isinstance(clf, _Transformer):
      clf = _Transformer(
        species,
        self.many_body_k,
        self.kbody_terms,
        split_dims=self._split_dims,
        order=self._order,
        periodic=self._periodic
      )
      self._transformers[formula] = clf
    return clf

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
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        y_true = atoms.get_total_energy()
        features, split_dims, _, multipliers, occurs = self.transform(
          species, atoms.get_positions(), cell, pbc,
        )

        x = _bytes_feature(features.tostring())
        y = _bytes_feature(np.atleast_2d(-y_true).tostring())
        z = _bytes_feature(occurs.tostring())
        w = _bytes_feature(multipliers.tostring())
        y_weight = _float_feature(loss_fn(y_true))
        example = Example(
          features=Features(feature={'energy': y, 'features': x, 'occurs': z,
                                     'weights': w, 'loss_weight': y_weight}))
        writer.write(example.SerializeToString())

        counter = Counter(species)
        for loc, atom in enumerate(self._ordered_species):
          coef[i, loc] = counter[atom]
        b[i] = y_true

        if verbose and i % 100 == 0:
          sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))

      if verbose:
        print("")
        print("Transforming %s finished!" % filename)

      return _compute_lr_weights(coef, b)

  def _save_auxiliary_for_file(self, filename, initial_weights=None,
                               indices=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      initial_weights: a 1D array of shape `[Nat, ]` as the initial weights for
        the one-body convolution kernel.
      indices: a `List[int]` as the indices of each given example.

    """
    name = splitext(basename(filename))[0]
    workdir = dirname(filename)
    cfgfile = join(workdir, "{}.json".format(name))
    indices = list(indices) if indices is not None else []
    weights = list(initial_weights) if initial_weights is not None else []
    max_occurs = {atom: times for atom, times in self._max_occurs.items()
                  if times < self._many_body_k}

    aux_dict = {
      "kbody_terms": self._kbody_terms,
      "split_dims": self._split_dims,
      "total_dim": self._total_dim,
      "lookup_indices": list([int(i) for i in indices]),
      "num_atom_types": len(self._ordered_species),
      "ordered_species": self._ordered_species,
      "two_body": self._num_ghosts > 0,
      "periodic": self._periodic,
      "many_body_k": self._many_body_k,
      "max_occurs": max_occurs,
      "order": self._order,
      "initial_one_body_weights": weights,
    }

    with open(cfgfile, "w+") as f:
      json.dump(aux_dict, fp=f, indent=2)

  def transform_and_save(self, db, train_file=None, test_file=None,
                         verbose=True, loss_fn=None):
    """
    Transform coordinates to input features and save them to tfrecord files
    using `tf.TFRecordWriter`.

    Args:
      db: a `Database` as the parsed results from a xyz file.
      train_file: a `str` as the file for saving training data or None to skip.
      test_file: a `str` as the file for saving testing data or None to skip.
      verbose: a `bool` indicating whether logging the transformation progress.
      loss_fn: a `Callable` for computing the exponential scaled RMSE loss.

    """
    if test_file:
      examples = db.examples(for_training=False)
      ids = db.ids_of_testing_examples
      num_examples = len(ids)
      self._transform_and_save(
        test_file, examples, num_examples, loss_fn=loss_fn, verbose=verbose)
      self._save_auxiliary_for_file(test_file, indices=ids)

    if train_file:
      examples = db.examples(for_training=True)
      ids = db.ids_of_training_examples
      num_examples = len(ids)
      weights = self._transform_and_save(
        train_file, examples, num_examples, loss_fn=loss_fn, verbose=verbose)
      self._save_auxiliary_for_file(
        train_file, initial_weights=weights, indices=ids)
