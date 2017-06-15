# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import time
import json
from kbody_transform import MultiTransformer
from kbody_input import extract_xyz
from save_model import get_tensors_to_restore
from os.path import join, dirname
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

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

  def _initialize_tensors(self):
    """
    Initialize the tensors.
    """
    tensors = {}
    for name, tensor_name in get_tensors_to_restore().items():
      tensors[name] = self._graph.get_tensor_by_name(tensor_name)

    # operations
    self._op_y_nn = tensors["Sum/1_and_k"]
    self._op_y_kbody = tensors["y_contribs"]
    self._op_y_1body = tensors["one-body/convolution"]

    # tensor
    self._t_1body = tensors["one-body/weights"]

    # placeholders
    self._pl_inputs = tensors["placeholders/inputs"]
    self._pl_occurs = tensors["placeholders/occurs"]
    self._pl_weights = tensors["placeholders/weights"]
    self._pl_split_dims = tensors["placeholders/split_dims"]
    self._pl_is_training = tensors["placeholders/is_training"]

  def _get_y_atomic_1body(self, species):
    """
    Return the 
    
    Args:
      species: a `List[str]` as the ordered species for this model.
    
    Returns:
      y_atomic_1body: a `Dict[str, float]` as the 1body energy for 
        each kind of atom.
    
    """
    weights = self._sess.run(self._t_1body)
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
      self._pl_inputs: features,
      self._pl_occurs: occurs,
      self._pl_weights: weights,
      self._pl_is_training: False,
      self._pl_split_dims: split_dims
    }

    # Run the operations to get the predicted energies.
    y_total, y_kbody, y_1body = self._sess.run(
      [self._op_y_nn, self._op_y_kbody, self._op_y_1body], feed_dict=feed_dict
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
  graph_model_path = join(
    dirname(__file__), "models", "C9H7N.PBE.v5", "C9H7N.PBE-1000000.pb")
  calculator = CNNPredictor(graph_model_path)
  elapsed = time.time() - tic
  print("Predictor initialized. Time: %.3f s" % elapsed)
  print("")

  print("------------")
  print("Tests: C9H7N")
  print("------------")

  xyzfile = join(dirname(__file__), "..", "datasets", "C9H7N.PBE.xyz")
  samples = extract_xyz(
    xyzfile, num_examples=5000, num_atoms=17, xyz_format='grendel')

  species = samples[0][0]
  y_true = float(samples[1][0])
  coords = samples[2][0]

  y_total, _, y_atomic, _ = calculator.predict(species, coords)
  _print_predictions(y_total, [y_true], y_atomic, species)


if __name__ == "__main__":
  tf.app.run(main=test)
