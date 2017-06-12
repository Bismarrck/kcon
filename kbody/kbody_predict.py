# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import time
import re
from kbody_transform import MultiTransformer, GHOST
from kbody_input import extract_xyz
from os.path import join, dirname
from collections import Counter

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class CNNPredictor:
  """
  An energy predictor based on the deep neural network of 'sum-kbody-cnn'.
  """

  def __init__(self, model_path, periodic=False, order=1, **kwargs):
    """
    Initialization method.

    Args:
      model_path: a `str` as the model to load. This `model_path` should contain 
        the model name and the global step, eg 'model.ckpt-500'.
      periodic: a `bool` indicating whether this model is used for periodic 
        structures or not.
      order: a `int` as the exponential order.
      kwargs: additional key-value arguments for importing the meta model.

    """
    self.sess = tf.Session()
    self._import_model(model_path, **kwargs)
    self._transformer = self._get_transformer(periodic=periodic, order=order)
    self._y_atomic_1body = self._get_y_atomic_1body(
      self._transformer.ordered_species)

  @property
  def many_body_k(self):
    """
    Return the many-body expansion factor for this model.
    """
    return self._transformer.many_body_k

  @property
  def is_periodic(self):
    """
    Return True if this model is used for periodic structures.
    """
    return self._transformer.is_periodic

  @staticmethod
  def _check_atom_types():
    """
    Automatically check the atom types and max occurances.
    """
    patt = re.compile(r'([A-Za-z]+)/k-Body/.*')
    kbody_terms = []
    for node in tf.get_default_graph().as_graph_def().node:
      m = patt.search(node.name)
      if m:
        kbody_terms.append(m.group(1))
    kbody_terms = set(kbody_terms)
    max_occurs = {}
    many_body_k = 0
    two_body = False
    if max(map(len, kbody_terms)) <= 2:
      many_body_k = 2
      atom_types = list(kbody_terms)
    else:
      max_occurs = Counter()
      for kbody_term in kbody_terms:
        atoms = []
        for atom in kbody_term:
          if atom.isupper():
            atoms.append(atom)
          else:
            atoms[-1] += atom
        for atom, n in Counter(atoms).items():
          max_occurs[atom] = max(max_occurs[atom], n)
        many_body_k = len(atoms)
      atom_types = list(max_occurs.keys())
      max_occurs = {k: v for k, v in max_occurs.items() if v < many_body_k}
      if GHOST in max_occurs:
        two_body = True
        del max_occurs[GHOST]
    if many_body_k < 2:
      raise Exception("The parsed `many_body_k` is invalid!")
    return many_body_k, atom_types, max_occurs, two_body

  def _import_model(self, model_path, **kwargs):
    """
    Import and restore the meta-model.

    Args:
      model_path: a `str` as the model to load. This `model_path` should contain 
        the model name and the global step, eg 'model.ckpt-500'.
      kwargs: additional key-value arguments for restoring the model.

    """
    self.saver = tf.train.import_meta_graph("{}.meta".format(model_path))
    self.saver.restore(self.sess, model_path, **kwargs)

    # Recover the tensors from the graph.
    graph = tf.get_default_graph()
    self.y_kbody_op = graph.get_tensor_by_name("Contribs:0")
    self.y_1body_op = graph.get_tensor_by_name("one-body/convolution:0")
    self._use_extra = graph.get_tensor_by_name(
      "placeholders/use_extra_inputs:0")
    self._extra_weights = graph.get_tensor_by_name(
      "placeholders/extra_weights:0")
    self._extra_inputs = graph.get_tensor_by_name("placeholders/extra_inputs:0")
    self._extra_occurs = graph.get_tensor_by_name("placeholders/extra_occurs:0")
    self._split_dims = graph.get_tensor_by_name("split_dims:0")
    self._shuffle_batch_inputs = graph.get_tensor_by_name(
      "input/shuffle_batch:0")
    self._shuffle_batch_occurs = graph.get_tensor_by_name(
      "input/shuffle_batch:2")
    self._shuffle_batch_weights = graph.get_tensor_by_name(
      "input/shuffle_batch:3")
    self._default_batch_input = np.zeros(
      self._shuffle_batch_inputs.get_shape().as_list(), dtype=np.float32)
    self._default_batch_occurs = np.ones(
      self._shuffle_batch_occurs.get_shape().as_list(), dtype=np.float32)
    self._default_batch_weights = np.zeros(
      self._shuffle_batch_weights.get_shape().as_list(), dtype=np.float32)

  def _get_transformer(self, periodic=False, order=1):
    """
    Return the feature transformer for this model.
    
    Args:
      periodic: a `bool` indicating whether this is a periodic model or not.
      order: a `int` as the exponential scaling order.

    Returns:
      transformer: a `MultiTransformer` for this model.

    """
    many_body_k, atom_types, max_occurs, two_body = self._check_atom_types()
    return MultiTransformer(
      atom_types,
      many_body_k=many_body_k,
      max_occurs=max_occurs,
      order=order,
      two_body=two_body,
      periodic=periodic,
    )

  def _get_y_atomic_1body(self, species):
    """
    Return the 
    
    Args:
      species: a `List[str]` as the ordered species for this model.
    
    Returns:
      y_atomic_1body: a `Dict[str, float]` as the 1body energy for 
        each kind of atom.
    
    """
    weights = self.sess.run(
      tf.get_default_graph().get_tensor_by_name("one-body/weights:0"))
    return dict(zip(species, weights.flatten().tolist()))

  def predict(self, species, coords, lattices=None, pbcs=None):
    """
    Make the prediction for the given molecule.

    Args:
      species: a `List[str]` as the ordered atomic species.
      coords: a 3D array of shape `[num_examples, num_atoms, 3]` as the atomic 
        coordinates.
      lattices: a 2D array of shape `[num_examples, 9]` as the periodic lattice
        matrix for each structure. Required if `self.is_periodic == True.`
      pbcs: a 2D boolean array of shape `[num_examples, 3]` as the periodic 
        conditions along XYZ. Required if `self.is_periodic == True.`

    Returns:
      y_total: a 1D array of shape `[num_examples, ]` as the total energies.
      y_1body: a 1D array of shape `[num_examples, ]` as the 1-body energies.
      y_atomics: a 2D array of shape `[num_examples, num_atoms]` as the 
        estimated atomic energies.
      y_kbody: a 2D array of shape `[num_examples, C(len(species), k)]` as the 
        kbody contribs.

    """

    # Check the shape of the `coords`.
    if len(coords.shape) == 2:
      assert coords.shape[0] == len(species)
      num_mols, num_atoms = 1, len(coords)
      coords = coords.reshape((1, num_atoms, 3))
    else:
      num_mols, num_atoms = coords.shape[0:2]
      assert num_atoms == len(species)

    # Check the lattices and pbcs for periodic structures.
    if self.is_periodic:
      assert (lattices is not None) and (pbcs is not None)

    # Transform the coordinates to input features. The `split_dims` will also be
    # returned.
    features, split_dims, _, weights, occurs = self._transformer.transform(
      species, coords, lattices=lattices, pbcs=pbcs
    )

    # Build the feed dict for running the session.
    features = features.reshape((num_mols, 1, -1, self._transformer.ck2))
    weights = weights.reshape((num_mols, 1, -1, 1))
    occurs = occurs.reshape((num_mols, 1, 1, -1))

    feed_dict = {
      # Make the model use `extra_inputs` as the input features.
      self._use_extra: True,
      self._extra_inputs: features,
      self._extra_weights: weights,
      self._extra_occurs: occurs,
      # This must be feeded but it will not be used.
      self._shuffle_batch_inputs: self._default_batch_input,
      self._shuffle_batch_weights: self._default_batch_weights,
      self._shuffle_batch_occurs: self._default_batch_occurs,
      # The dimensions for splitting the input features.
      self._split_dims: split_dims
    }

    # Run the operations to get the predicted energies.
    y_kbody_raw, y_1body_raw = self.sess.run(
      [self.y_kbody_op, self.y_1body_op], feed_dict=feed_dict
    )

    # Compute the total energies
    y_1body = np.squeeze(y_1body_raw)
    y_kbody = np.multiply(y_kbody_raw, weights)
    y_total = np.squeeze(np.sum(y_kbody, axis=2, keepdims=True)) + y_1body

    # Transform the kbody energies to atomic energies.
    y_kbody = np.squeeze(y_kbody, axis=(1, 3))
    y_atomic = self._transformer.compute_atomic_energies(
      species, y_kbody, self._y_atomic_1body)

    return (np.negative(np.atleast_1d(y_total)),
            np.negative(np.atleast_1d(y_1body)),
            np.negative(y_atomic),
            np.negative(y_kbody))


def _print_predictions(y_total, y_true, y_atomic, species):
  """
  A helper function for printing predicted results of the unittests.

  Args:
    y_total: a 1D array of shape [N, ] as the predicted energies. 
    y_true: a 1D array of shape [N, ] as the real energies.
    y_atomic: a 2D array of shape [N, M] as the atomic energies.
    species: a `List[str]` as the atomic species.

  """
  num_examples, num_atoms = y_atomic.shape
  size = min(num_examples, 20)
  y_total = np.atleast_1d(y_total)
  y_true = np.atleast_1d(y_true)
  for i in np.random.choice(range(num_examples), size=size):
    print("Index            : % 2d" % i)
    print("Energy Predicted : % .4f eV" % y_total[i])
    print("Energy Real      : % .4f eV" % y_true[i])
    for j in range(num_atoms):
      print("Atom %2d, %2s,     % 10.4f eV" % (j, species[j], y_atomic[i, j]))
    print("")


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def test(unused):

  tic = time.time()
  model_path = join(
    dirname(__file__), "models", "TiO2.v5.k123", "model.ckpt-1733772")
  calculator = CNNPredictor(model_path, periodic=True)
  elapsed = time.time() - tic
  print("Predictor initialized. Time: %.3f s" % elapsed)
  print("")

  print("-----------")
  print("Tests: TiO2")
  print("-----------")

  xyzfile = join(dirname(__file__), "..", "test_files", "TiO2_x9_sample.xyz")
  sample = extract_xyz(
    xyzfile, num_examples=1, num_atoms=27, xyz_format='grendel')

  species = sample[0][0]
  y_true = float(sample[1])
  coords = sample[2]
  lattice = sample[4]
  pbc = sample[5]
  y_total, _, y_atomic, _ = calculator.predict(species, coords, lattice, pbc)
  _print_predictions(y_total, [y_true], y_atomic, species)


if __name__ == "__main__":
  tf.app.run(main=test)
