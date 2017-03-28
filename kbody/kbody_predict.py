# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody.kbody_transform as kbody_transform

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class CNNPredictor:
  """
  An energy predictor based on the deep neural network of 'sum-kbody-cnn'.
  """

  def __init__(self, atom_types, model_path, many_body_k=3, max_occurs=None):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the target atomic species.
      model_path: a `str` as the model to load. This `model_path` should contain 
        the model name and the global step, eg 'model.ckpt-500'.
      many_body_k: a `int` as the many body expansion factor.
      max_occurs: a `Dict[str, int]` as the maximum appearance for a specie.

    """
    self.transformer = kbody_transform.MultiTransformer(
      atom_types,
      many_body_k=many_body_k,
      max_occurs=max_occurs,
    )
    self.sess = tf.Session()
    self._import_model(model_path)

  @property
  def many_body_k(self):
    """
    Return the many-body expansion factor for this model.
    """
    return self.transformer.many_body_k

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
    self.y_total_op = graph.get_tensor_by_name("Outputs/squeeze:0")
    self.y_kbody_op = graph.get_tensor_by_name("Contribs:0")
    self._extra_inputs = graph.get_tensor_by_name("placeholders/extra_inputs:0")
    self._use_extra = graph.get_tensor_by_name(
      "placeholders/use_extra_inputs:0")
    self._is_predicting = graph.get_tensor_by_name(
      "placeholders/is_predicting:0")
    self._split_dims = graph.get_tensor_by_name("split_dims:0")
    self._shuffle_batch = graph.get_tensor_by_name("input/shuffle_batch:0")
    self._defaut_batch = np.zeros(self._shuffle_batch.get_shape().as_list(),
                                  dtype=np.float32)

  def predict(self, species, coords):
    """
    Make the prediction for the given molecule.

    Args:
      species: a `List[str]` as the ordered atomic species of a molecule.
      coords: a 2D array as the atomic coordinates of a molecule.

    Returns:
      total_energy: a float as the predicted total energy.
      atomic_energies: a 1D array as the predicted energy for each atom.
      kbody_energies: a 1D array as the predicted energy of each k-body terms.

    """

    # Check the shape of the `coords`.
    if len(coords.shape) == 2:
      assert coords.shape[0] == len(species)
      num_mols, num_atoms = 1, len(coords)
      coords = coords.reshape((1, num_atoms, 3))
    else:
      num_mols, num_atoms = coords.shape[0:2]
      assert num_atoms == len(species)

    # Transform the coordinates to input features. The `split_dims` will also be
    # returned.
    features, split_dims, _ = self.transformer.transform(species, coords)

    # Build the feed dict for running the session.
    features = features.reshape((num_mols, 1, -1, self.ck2))
    feed_dict = {
      # Make the model use `extra_inputs` as the input features.
      self._use_extra: True,
      self._extra_inputs: features,
      # The zero-padding check is enabled so that we can predict AxByHz systems.
      self._is_predicting: True,
      # This must be feeded but it will not be used.
      self._shuffle_batch: self._defaut_batch,
      # The dimensions for splitting the input features.
      self._split_dims: split_dims
    }

    # Run the operations to get the predicted energies.
    y_total, y_kbody = sess.run(
      [self.y_total_op, self.y_kbody_op],
      feed_dict=feed_dict
    )

    # Transform the kbody energies to atomic energies.
    y_atomic = self.transformer.compute_atomic_energies(species, y_kbody)

    return np.negative(y_total), np.negative(y_atomic)
