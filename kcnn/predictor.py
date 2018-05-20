# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import json
from collections import Counter
from ase import Atoms
from ase.io.trajectory import Trajectory
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from constants import GHOST
from save_model import get_tensors_to_restore
from transformer import MultiTransformer, FixedLenMultiTransformer

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def restore_transformer(graph, session, fixed=False):
  """
  Restore a `MultiTransformer` from the freezed graph.

  Args:
    graph: a `tf.Graph`.
    session: a `tf.Session` to execute ops.
    fixed: a `bool`. If True, a `FixedLenMultiTransformer` will be restored.
      Otherwise a `MultiTransformer` will be restored.

  Returns:
    clf: a `MultiTransformer` or a `FixedLenMultiTransformer`.

  """
  tensor = graph.get_tensor_by_name("transformer/json:0")
  params = dict(json.loads(session.run(tensor).decode()))
  if not fixed:
    return MultiTransformer(**{k: v for k, v in params.items()
                               if k != "species"})
  else:
    max_occurs = Counter(params["species"])
    kwargs = {k: v for k, v in params.items()
              if k not in ("species", "atom_types", "max_occurs")}
    return FixedLenMultiTransformer(max_occurs, **kwargs)


class KcnnPredictor:
  """
  An energy predictor based on the deep neural network of 'KCNN'.
  """

  def __init__(self, graph_model_path, fixed=False):
    """
    Initialization method.

    Args:
      graph_model_path: a `str` as the freezed graph model to load.
      fixed: a `bool`. If True, a `FixedLenMultiTransformer` will be restored.
        Otherwise a `MultiTransformer` will be restored.

    """

    graph = tf.Graph()

    with graph.as_default():
      output_graph_def = graph_pb2.GraphDef()
      with open(graph_model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        importer.import_graph_def(output_graph_def, name="")

    self._graph = graph
    self._sess = tf.Session(graph=graph)
    self._transformer = restore_transformer(self._graph, self._sess, fixed)
    assert isinstance(self._transformer, MultiTransformer)

    self._initialize_tensors()
    self._y_atomic_1body = self._get_y_atomic_1body(
      self._transformer.atom_types
    )

  @property
  def k_max(self):
    """
    Return the many-body expansion factor for this model.
    """
    return self._transformer.k_max

  @property
  def included_k(self):
    """
    Return the included k under the many body expansion scheme.
    """
    return self._transformer.included_k

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

  @property
  def atomic_forces(self):
    """
    Return True if this predictor supports predicting atomic forces.

    Notes:
      Currently the atomic forces are not supported!

    """
    return self._transformer.atomic_forces_enabled

  @property
  def transformer(self):
    """
    Return the `MultiTransformer` of this predictor.
    """
    return self._transformer

  def __str__(self):
    """
    Return the string representation of this predictor.
    """
    species = list(self._transformer.species)
    if GHOST in species:
      species.remove(GHOST)
    if not self.is_periodic:
      outputs = ["KcnnPredictor of {}".format("".join(species))]
    else:
      outputs = ["Periodic KcnnPredictor of {}".format("".join(species))]
    for atom, (nmin, nmax) in self.supported_stoichiometries.items():
      outputs.append("  {:2s} : [{}, {}]".format(atom, nmin, nmax))
    outputs.append("End")
    return "\n".join(outputs)

  def _initialize_tensors(self):
    """
    Initialize the tensors.
    """
    tensors = {}
    forces = self.atomic_forces
    for name, tensor_name in get_tensors_to_restore(forces=forces).items():
      tensors[name] = self._graph.get_tensor_by_name(tensor_name)

    # The operators
    self._operator_y_nn = tensors["kCON/Energy/Sum/1_and_k"]
    self._operator_y_kbody = tensors["kCON/Energy/y_contribs"]
    self._operator_y_1body = tensors["kCON/Energy/one-body/Conv2D"]

    # The data tensor
    self._tensor_1body = tensors["kCON/one-body/weights"]

    # Placeholders
    self._placeholder_inputs = tensors["placeholders/inputs"]
    self._placeholder_occurs = tensors["placeholders/occurs"]
    self._placeholder_weights = tensors["placeholders/weights"]
    self._placeholder_split_dims = tensors["placeholders/split_dims"]

    # Tensors for predicting atomic forces
    if self._transformer.atomic_forces_enabled:
      self._placeholder_coefficients = tensors["placeholders/coefficients"]
      self._placeholder_indexing = ["placeholders/indexing"]
      self._operator_f_nn = tensors["kCON/Forces/forces"]
    else:
      self._placeholder_coefficients = None
      self._placeholder_indexing = None
      self._operator_f_nn = None

    # Temporarily disable this tensor. I think this will be removed soon.
    # self._placeholder_is_training = tensors["placeholders/is_training"]

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

  def get_feed_dict(self, atoms_or_trajectory):
    """
    Return the feed dict for the inputs.

    Args:
      atoms_or_trajectory: an `ase.Atoms` or an `ase.io.TrajectoryReader` or a
        list of `ase.Atoms` with the same stoichiometry.

    Returns:
      species: a list of `str` as the stoichiometry.
      feed_dict: a `dict` as the feed dict to run.

    """
    assert isinstance(self._transformer, MultiTransformer)

    if isinstance(atoms_or_trajectory, Atoms):
      transform_func = self._transformer.transform
      ntotal = 1
      species = atoms_or_trajectory.get_chemical_symbols()
    elif isinstance(atoms_or_trajectory, (list, tuple, Trajectory)):
      transform_func = self._transformer.transform_trajectory
      ntotal = len(atoms_or_trajectory)
      species = atoms_or_trajectory[0].get_chemical_symbols()
    else:
      raise ValueError("`atoms_or_trajectory` should be an `ase.Atoms` or an"
                       "`ase.io.TrajectoryReader` or a list of `ase.Atoms`!")

    # The `transformed` is a named tupe: KcnnSample.
    transformed = transform_func(atoms_or_trajectory)

    # Build the feed dict for running the session.
    ck2 = self._transformer.ck2
    features = transformed.features.reshape((ntotal, 1, -1, ck2))
    weights = transformed.binary_weights.reshape((ntotal, 1, -1, 1))
    occurs = transformed.occurs.reshape((ntotal, 1, 1, -1))
    split_dims = transformed.split_dims

    feed_dict = {self._placeholder_inputs: features,
                 self._placeholder_occurs: occurs,
                 self._placeholder_weights: weights,
                 self._placeholder_split_dims: split_dims}
    return species, feed_dict

  def predict_total_energy(self, atoms_or_trajectory):
    """
    Only make predictions of total energies. All input structures must have the
    same kind of atomic species.

    Args:
      atoms_or_trajectory: an `ase.Atoms` or an `ase.io.TrajectoryReader` or a
        list of `ase.Atoms` with the same stoichiometry.

    Returns:
      y_total: a 1D array of shape `[num_examples, ]` as the total energies.

    """
    _, feed_dict = self.get_feed_dict(atoms_or_trajectory)
    y_total = self._sess.run(self._operator_y_nn, feed_dict=feed_dict)
    return np.negative(y_total)

  def predict(self, atoms_or_trajectory):
    """
    Make the prediction for the given structures. All input structures must have
    the same kind of atomic species.

    Args:
      atoms_or_trajectory: an `ase.Atoms` or an `ase.io.TrajectoryReader` or a
        list of `ase.Atoms` with the same stoichiometry.

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
    species, feed_dict = self.get_feed_dict(atoms_or_trajectory)

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
