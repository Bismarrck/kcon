# coding=utf-8
"""
This script is used to transform atomic coordinates to input features and then
save them in tfrecords files.
"""

from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import sys
import json
from scipy.misc import comb
from itertools import combinations, product, repeat, chain
from sklearn.metrics import pairwise_distances
from collections import Counter
from os.path import basename, dirname, join, splitext, isfile
from os import remove
from tensorflow.python.training.training import Features, Example

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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
  'V': 1.34, 'W': 1.37, 'X': 0.32, 'Xe': 1.31, 'Y': 1.63, 'Yb': 1.7,
  'Zn': 1.18, 'Zr': 1.54
}

flags = tf.app.flags

FLAGS = flags.FLAGS


def get_formula(species):
  """
  Return the molecular formula given a list of atomic species.
  """
  return "".join(species)


def _get_pyykko_bonds_matrix(species, factor=1.0, flatten=True):
  """
  Return the pyykko-bonds matrix given a list of atomic symbols.

  Args:
    species: List[str], a list of atomic symbols.
    factor: a float, the scaling factor.
    flatten: a bool. The bonds matrix will be flatten to a 1D array if True.

  Returns:
    bonds: the bonds matrix (or vector if `flatten` is True).

  """
  rr = np.asarray([pyykko[specie] for specie in species])[:, np.newaxis]
  lmat = np.multiply(factor, rr + rr.T)
  if flatten:
    return lmat.flatten()
  else:
    return lmat


def _gen_dist2inputs_mapping(species, kbody_terms):
  """
  Build the mapping from interatomic distances matrix to the [C(N,k), C(k,2)]
  feature matrix.

  Args:
    species: a list of str as the ordered atomic symbols.
    kbody_terms: a list of comma-separated elements string as the ordered k-body
      atomic symbol combinations.

  Returns:
    mapping: a `Dict[str, np.ndarray]` as the mapping from the N-by-N distance 
      matrix to the feature matrix for each k-body term.
    selection: a `Dict[str, List[List[int]]]` as the corresponding atom indices 
      for each k-body term.

  """
  natoms = len(species)
  uniques = set(species)
  element_indices = {}
  for element in uniques:
    for i in range(natoms):
      if species[i] == element:
        element_indices[element] = element_indices.get(element, []) + [i]
  mapping = {}
  selections = {}
  for term in kbody_terms:
    elements = term.split(",")
    counter = Counter(elements)
    # If the occurances of an element in the k-body term is bigger than that in
    # the species, we shall discard this kbody term.
    if any([counter[e] > len(element_indices.get(e, [])) for e in elements]):
      continue
    ck2 = comb(len(elements), 2, exact=True)
    keys = sorted(counter.keys())
    kbodies = [[list(o) for o in combinations(element_indices[e], counter[e])]
               for e in keys]
    # All k-body combinations
    kbodies = [list(chain(*o)) for o in product(*kbodies)]
    selections[term] = kbodies
    cnk = len(kbodies)
    mapping[term] = np.zeros((ck2, cnk), dtype=int)
    for i in range(cnk):
      for j, (vi, vj) in enumerate(combinations(kbodies[i], 2)):
        mapping[term][j, i] = vi * natoms + vj
  return mapping, selections


def _gen_sorting_indices(kbody_terms):
  """
  Generate the soring indices.

  Args:
    kbody_terms: a `List[str]` as the ordered k-body terms.

  Returns:
    indices: a dict of indices for sorting along the last axis of the input
      features.

  """
  indices = {}
  for term in kbody_terms:
    elements = list(sorted(term.split(",")))
    atom_pairs = list(combinations(elements, r=2))
    n = len(atom_pairs)
    counter = Counter(atom_pairs)
    if max(counter.values()) == 1:
      continue
    indices[term] = []
    for pair, times in counter.items():
      if times > 1:
        indices[term].append([i for i in range(n) if atom_pairs[i] == pair])
  return indices


def exponential(inputs, factor, order=1):
  """
  Exponentially scale the `inputs`.

  Args:
    inputs: Union[float, np.ndarray] as the inputs to scale.
    factor: a float or an array with the same shape of `inputs` as the scaling 
      factor(s).
    order: a `int` as the exponential order.

  Returns:
    scaled: the scaled inputs.

  """
  return np.exp(np.negative((inputs / factor)**order))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Transformer:
  """
  This class is used to transform atomic coordinates and energies to input
  features and training targets.
  """

  def __init__(self, species, many_body_k=4, kbody_terms=None, split_dims=None,
               order=1):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      many_body_k: a `int` as the maximum order for the many-body expansion.
      kbody_terms: a `List[str]` as the k-body terms.
      split_dims: a `List[int]` as the dimensions for splitting inputs. If this
        is given, the `kbody_terms` must also be set and their lengths should be
        equal.

    """
    if split_dims is not None:
      assert len(split_dims) == len(kbody_terms)

    if kbody_terms is None:
      kbody_terms = sorted(list(set(
        [",".join(sorted(c)) for c in combinations(species, many_body_k)])))

    # Generate the mapping from the N-by-N interatomic distances matrix to the
    # [C(N,k), C(k,2)] input feature matrix and the indices selections for each
    # C(N,k) terms.
    mapping, selections = _gen_dist2inputs_mapping(species, kbody_terms)

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
    self._sorting_indices = _gen_sorting_indices(kbody_terms)
    self._lmat = _get_pyykko_bonds_matrix(species)
    self._kbody_sizes = kbody_sizes
    self._multipliers = multipliers
    self._order = order

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

  def transform(self, coordinates, energies=None):
    """
    Transform the given atomic coordinates and energies to input features and
    training targets and return them as numpy arrays.

    Args:
      coordinates: a 3D array as the atomic coordinates of sturctures.
      energies: a 1D array as the desired energies.

    Returns:
      features: a 4D array as the transformed input features.
      targets: a 1D array as the training targets (actually the negative of the
        input energies) given `energies` or None.

    """
    num_examples = len(coordinates)
    samples = np.zeros((num_examples, self._total_dim, self.ck2),
                       dtype=np.float32)
    kbody_terms = self._kbody_terms
    mapping = self._dist2inputs_mapping
    offsets = self._kbody_offsets

    for i in range(num_examples):
      dists = pairwise_distances(coordinates[i]).flatten()
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

    if energies is not None:
      targets = np.negative(energies)
    else:
      targets = None
    return samples, targets


class MultiTransformer:
  """
  A flexible transformer targeting on AxByCz ... molecular compositions.
  """

  def __init__(self, atom_types, many_body_k=3, max_occurs=None, order=1):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the target atomic species.  
      many_body_k: a `int` as the many body expansion factor.
      max_occurs: a `Dict[str, int]` as the maximum appearances for a specie. 
        If an atom is explicitly specied, it can appear infinity times.

    """
    self._many_body_k = many_body_k
    self._atom_types = list(set(atom_types))
    species = []
    max_occurs = {} if max_occurs is None else dict(max_occurs)
    for specie in self._atom_types:
      species.extend(list(repeat(specie, max_occurs.get(specie, many_body_k))))
      if specie not in max_occurs:
        max_occurs[specie] = np.inf
    self._kbody_terms = sorted(list(
        set([",".join(sorted(c)) for c in combinations(species, many_body_k)])))
    self._transformers = {}
    self._max_occurs = max_occurs
    self._order = order

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
    formula = get_formula(species)
    clf = self._transformers.get(formula)
    if not isinstance(clf, Transformer):
      clf = Transformer(
        species,
        self.many_body_k,
        kbody_terms=self.kbody_terms,
        order=self._order
      )
      self._transformers[formula] = clf
    return clf

  def transform(self, species, coords, energies=None):
    """
    Transform the atomic coordinates to input features.

    Args:
      species: a `List[str]` as the ordered atomic species for molecule(s).
      coords: a 2D or 3D array as the atomic coordinates. If this is a 2D array, 
        it should represents the N-by-3 coordinates of a single molecule.
      energies: a 1D array as the total energies. 

    Returns:
      features: a 4D array as the input features.
      split_dims: a `List[int]` for splitting the input features along axis 2.
      targets: a 1D array as the training targets given `energis` or None.
      weights: a 1D array as the binary weights of each row of the features. 
    
    Raises:
      ValueError: if the `species` is not supported by this transformer.

    """
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    coords = np.asarray(coords)
    if len(coords.shape) == 2:
      if coords.shape[0] != len(species):
        raise ValueError("The shapes of coords and species are not matched!")
      coords = coords.reshape((1, len(species), 3))
    elif coords.shape[1] != len(species):
      raise ValueError("The shapes of coords and species are not matched!")

    clf = self._get_transformer(species)
    features, targets = clf.transform(coords, energies)
    return features, clf.split_dims, targets, clf.multipliers

  def compute_atomic_energies(self, species, kbody_contribs):
    """
    Compute the atomic energies given predicted kbody contributions.
    
    Args:
      species: a `List[str]` as the ordered atomic species for molecules.
      kbody_contribs: a 2D array as the kbody contributions of each molecule.

    Returns:
      atomic_energies: a 2D array as the atomic energies of each molecule.

    """
    clf = self._get_transformer(species)
    split_dims = clf.split_dims
    num_mols = kbody_contribs.shape[0]
    num_atoms = len(species)
    atomic_energies = np.zeros((num_mols, num_atoms))
    for step in range(num_mols):
      for i, term in enumerate(clf.kbody_terms):
        if term not in clf.kbody_selections:
          continue
        istart = 0 if i == 0 else int(sum(split_dims[:i]))
        for kbody_i, indices in enumerate(clf.kbody_selections[term]):
          for atom_i in indices:
            atomic_energies[step, atom_i] += \
              kbody_contribs[step, istart + kbody_i]
    return atomic_energies / float(self.ck2)


class FixedLenMultiTransformer(MultiTransformer):
  """
  This is also a flexible transformer targeting on AxByCz ... molecular 
  compositions but the length of the transformed features are the same so that
  they can be used for training.
  """

  def __init__(self, max_occurs, many_body_k=3, order=1):
    """
    Initialization method. 
    
    Args:
      max_occurs: a `Dict[str, int]` as the maximum appearances for each kind of 
        atomic specie. 
      many_body_k: a `int` as the many body expansion factor.
    
    """
    super(FixedLenMultiTransformer, self).__init__(
      list(max_occurs.keys()),
      many_body_k=many_body_k,
      max_occurs=max_occurs,
      order=order,
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
    formula = get_formula(species)
    clf = self._transformers.get(formula)
    if not isinstance(clf, Transformer):
      clf = Transformer(
        species,
        self.many_body_k,
        self.kbody_terms,
        split_dims=self._split_dims,
        order=self._order
      )
      self._transformers[formula] = clf
    return clf

  def _transform_and_save(self, array_of_species, energies, coords, filename,
                          verbose=True):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      array_of_species: a `List[List[str]]` as the species of the isomers.
      energies: a 1D array as the desired energies.
      coords: a 3D array as the atomic coordinates of sturctures.
      filename: a `str` as the file to save examples.
      verbose: boolean indicating whether.

    """

    with tf.python_io.TFRecordWriter(filename) as writer:

      if verbose:
        print("Start mixed transforming %s ... " % filename)

      num_examples = len(array_of_species)

      for i in range(len(array_of_species)):
        species = array_of_species[i]
        n = len(species)
        features, split_dims, targets, multipliers = self.transform(
          species,
          coords[i, :n]
        )
        x = _bytes_feature(features.tostring())
        y = _bytes_feature(np.atleast_2d(-1.0 * energies[i]).tostring())
        w = _bytes_feature(multipliers.tostring())

        example = Example(
          features=Features(feature={
            'energy': y,
            'features': x,
            'weights': w
          }))
        writer.write(example.SerializeToString())

        if verbose and i % 100 == 0:
          sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))

      if verbose:
        print("")
        print("Transforming %s finished!" % filename)

  def _save_auxiliary_for_file(self, filename, indices=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      indices: a `List[int]` as the indices of each given example.

    """
    name = splitext(basename(filename))[0]
    workdir = dirname(filename)
    cfgfile = join(workdir, "{}.json".format(name))
    if indices is not None:
      indices = list(indices)
    else:
      indices = []

    with open(cfgfile, "w+") as f:
      json.dump({
        "kbody_terms": self._kbody_terms,
        "split_dims": self._split_dims,
        "total_dim": self._total_dim,
        "inverse_indices": list([int(i) for i in indices])
      }, f, indent=2)

  def transform_and_save(self, array_of_species, energies, coords, filename,
                         indices=None, verbose=True):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      array_of_species: a `List[List[str]]` as the species of the isomers.
      coords: a 3D array as the atomic coordinates of sturctures.
      energies: a 1D array as the desired energies.
      filename: a `str` as the file to save examples.
      indices: a `List[int]` as the indices of each given example. This is an
        optional argument.
      verbose: a `bool` indicating whether logging the transformation progress.

    """
    try:
      self._transform_and_save(
        array_of_species, energies, coords, filename, verbose=verbose)
    except Exception as excp:
      if isfile(filename):
        remove(filename)
      raise excp
    else:
      self._save_auxiliary_for_file(filename, indices=indices)
