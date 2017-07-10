# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import json

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from kbody_transform import MultiTransformer
from constants import GHOST
from save_model import get_tensors_to_restore

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def restore_transformer(graph, session):
  """
  Restore a `MultiTransformer` from the freezed graph.

  Args:
    graph: a `tf.Graph`.
    session: a `tf.Session` to execute ops.

  Returns:
    clf: a `MultiTransformer` restored from the graph.

  """
  tensor = graph.get_tensor_by_name("transformer/json:0")
  params = dict(json.loads(session.run(tensor).decode()))
  return MultiTransformer(**params)


class CNNPredictor:
  """
  An energy predictor based on the deep neural network of 'sum-kbody-cnn'.
  """

  def __init__(self, graph_model_path):
    """
    Initialization method.

    Args:
      graph_model_path: a `str` as the freezed graph model to load.

    """

    graph = tf.Graph()

    with graph.as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(graph_model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        importer.import_graph_def(output_graph_def, name="")

    self._graph = graph
    self._sess = tf.Session(graph=graph)
    self._initialize_tensors()
    self._transformer = restore_transformer(self._graph, self._sess)
    self._y_atomic_1body = self._get_y_atomic_1body(
      self._transformer.ordered_species
    )

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

  @property
  def sess(self):
    """
    Return the associated session for this predictor.
    """
    return self._sess

  @property
  def graph(self):
    """
    Return the restored freezed graph.
    """
    return self._graph

  @property
  def supported_stoichiometries(self):
    """
    Return the supported stoichiometries.
    """
    supported = {}
    max_occurs = self._transformer.max_occurs
    for atom, max_occur in max_occurs.items():
      if atom != GHOST:
        supported[atom] = [0, max_occur]
    return supported

  @property
  def kbody_terms(self):
    """
    Return the associated kbody terms.
    """
    return self._transformer.kbody_terms

  def __str__(self):
    """
    Return the string representation of this predictor.
    """
    species = list(self._transformer.ordered_species)
    if GHOST in species:
      species.remove(GHOST)
    if not self.is_periodic:
      outputs = ["CNNPredictor of {}".format("".join(species))]
    else:
      outputs = ["Periodic CNNPredictor of {}".format("".join(species))]
    for atom, (nmin, nmax) in self.supported_stoichiometries.items():
      outputs.append("  {:2s} : [{}, {}]".format(atom, nmin, nmax))
    outputs.append("End")
    return "\n".join(outputs)

  def _initialize_tensors(self):
    """
    Initialize the tensors.
    """
    tensors = {}
    for name, tensor_name in get_tensors_to_restore().items():
      tensors[name] = self._graph.get_tensor_by_name(tensor_name)

    # operations
    self._operator_y_nn = tensors["Sum/1_and_k"]
    self._operator_y_kbody = tensors["y_contribs"]
    self._operator_y_1body = tensors["one-body/convolution"]

    # tensor
    self._tensor_1body = tensors["one-body/weights"]

    # placeholders
    self._placeholder_inputs = tensors["placeholders/inputs"]
    self._placeholder_occurs = tensors["placeholders/occurs"]
    self._placeholder_weights = tensors["placeholders/weights"]
    self._placeholder_split_dims = tensors["placeholders/split_dims"]
    self._placeholder_is_training = tensors["placeholders/is_training"]

  def _get_y_atomic_1body(self, species):
    """
    Return the one-body energy for each type of atom.
    
    Args:
      species: a `List[str]` as the ordered species for this model.
    
    Returns:
      y_atomic_1body: a `Dict[str, float]` as the 1body energy for 
        each kind of atom.
    
    """
    weights = self._sess.run(self._tensor_1body)
    return dict(zip(species, weights.flatten().tolist()))

  def _check_inputs(self, species, array_of_coords, array_of_lattice,
                    array_of_pbc):
    """
    Check the inputs before making predictions.
    """

    if len(array_of_coords.shape) == 2:
      assert array_of_coords.shape[0] == len(species)
      num_mols, num_atoms = 1, len(array_of_coords)
      array_of_coords = array_of_coords.reshape((1, num_atoms, 3))
      if array_of_lattice is not None:
        if len(array_of_lattice.shape) == 1:
          array_of_lattice = array_of_lattice.reshape((1, -1))
        if len(array_of_pbc.shape) == 1:
          array_of_pbc = array_of_pbc.reshape((1, -1))
    else:
      num_mols, num_atoms = array_of_coords.shape[0:2]
      assert num_atoms == len(species)

    if self.is_periodic:
      assert (array_of_lattice is not None) and (array_of_pbc is not None)

    return num_mols, {"array_of_coords": array_of_coords,
                      "array_of_lattice": array_of_lattice,
                      "array_of_pbc": array_of_pbc}

  def get_feed_dict(self, species, array_of_coords, array_of_lattice=None,
                    array_of_pbc=None):
    """
    Return the feed dict for the inputs.

    Args:
      species: a `List[str]` as the ordered atomic species.
      array_of_coords: a `float32` array of shape `[num_examples, num_atoms, 3]`
        as the atomic coordinates.
      array_of_lattice: a `float32` array of shape `[num_examples, 9]` as the
        periodic cell parameters for each structure.
      array_of_pbc: a `bool` array of shape `[num_examples, 3]` as the periodic
        conditions along XYZ directions.

    Returns:
      feed_dict: a `dict` that should be feeded to the session.

    """
    ntotal, inputs = self._check_inputs(
      species, array_of_coords, array_of_lattice, array_of_pbc
    )

    # Transform the coordinates to input features. The `split_dims` will also be
    # returned.
    features, split_dims, _, weights, occurs = self._transformer.transform(
      species, **inputs
    )

    # Build the feed dict for running the session.
    features = features.reshape((ntotal, 1, -1, self._transformer.ck2))
    weights = weights.reshape((ntotal, 1, -1, 1))
    occurs = occurs.reshape((ntotal, 1, 1, -1))

    return {self._placeholder_inputs: features,
            self._placeholder_occurs: occurs,
            self._placeholder_weights: weights,
            self._placeholder_is_training: False,
            self._placeholder_split_dims: split_dims}

  def predict_total(self, species, array_of_coords, array_of_lattice=None,
                    array_of_pbc=None):
    """
    Only make predictions of total energies. All input structures must have the
    same kind of atomic species.

    Args:
      species: a `List[str]` as the ordered atomic species.
      array_of_coords: a `float32` array of shape `[num_examples, num_atoms, 3]`
        as the atomic coordinates.
      array_of_lattice: a `float32` array of shape `[num_examples, 9]` as the
        periodic cell parameters for each structure.
      array_of_pbc: a `bool` array of shape `[num_examples, 3]` as the periodic
        conditions along XYZ directions.

    Returns:
      y_total: a 1D array of shape `[num_examples, ]` as the total energies.

    """
    feed_dict = self.get_feed_dict(species, array_of_coords,
                                   array_of_lattice, array_of_pbc)
    y_total = self._sess.run(self._operator_y_nn, feed_dict=feed_dict)
    return np.negative(y_total)

  def predict(self, species, array_of_coords, array_of_lattice=None,
              array_of_pbc=None):
    """
    Make the prediction for the given structures. All input structures must have
    the same kind of atomic species.

    Args:
      species: a `List[str]` as the ordered atomic species.
      array_of_coords: a `float32` array of shape `[num_examples, num_atoms, 3]`
        as the atomic coordinates.
      array_of_lattice: a `float32` array of shape `[num_examples, 9]` as the
        periodic cell parameters for each structure.
      array_of_pbc: a `bool` array of shape `[num_examples, 3]` as the periodic
        conditions along XYZ directions.

    Returns:
      y_total: a `float32` array of shape `[num_examples, ]` as the predicted
        total energies.
      y_1body: a `float32` array of shape `[num_examples, ]` as the predicted
        one-body energies.
      y_atomics: a `float32` array of shape `[num_examples, num_atoms]` as the
        predicted energies for atoms which sum up to total energies.
      y_kbody: a `float32` array of shape `[num_examples, D]` as the kbody
        contribs. `D` is the total dimension.

    """
    feed_dict = self.get_feed_dict(species, array_of_coords,
                                   array_of_lattice, array_of_pbc)

    # Run the operations to get the predicted energies.
    y_total, y_kbody, y_1body = self._sess.run(
      [self._operator_y_nn, self._operator_y_kbody, self._operator_y_1body],
      feed_dict=feed_dict
    )
    y_1body = np.squeeze(y_1body)

    # Compute the atomic energies from kbody contribs.
    y_kbody = np.squeeze(y_kbody, axis=(1, 3))
    y_atomic = self._transformer.compute_atomic_energies(
      species, y_kbody, self._y_atomic_1body)

    return (np.negative(np.atleast_1d(y_total)),
            np.negative(np.atleast_1d(y_1body)),
            np.negative(y_atomic),
            np.negative(y_kbody))

  def eval(self, name, feed_dict=None):
    """
    Evaluate a specific tensor from the graph given the feed dict.

    Args:
      name: a `str` as the name of the tensor.
      feed_dict: a `dict` as the feed dict.

    Returns:
      results: a `object` as the evaluation result.

    """
    try:
      tensor = self._graph.get_tensor_by_name(name)
    except Exception:
      raise KeyError(
        "The tensor {} can not be found in the graph!".format(name))
    else:
      return self.sess.run(tensor, feed_dict=feed_dict)
