# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import time
from kbody_transform import MultiTransformer
from kbody_input import hartree_to_ev
from os.path import join, dirname
from itertools import repeat

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class CNNPredictor:
  """
  An energy predictor based on the deep neural network of 'sum-kbody-cnn'.
  """

  def __init__(self, atom_types, model_path, many_body_k=3, max_occurs=None,
               order=1, **kwargs):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the target atomic species.
      model_path: a `str` as the model to load. This `model_path` should contain 
        the model name and the global step, eg 'model.ckpt-500'.
      many_body_k: a `int` as the many body expansion factor.
      max_occurs: a `Dict[str, int]` as the maximum appearance for a specie.
      kwargs: additional key-value arguments for importing the meta model.

    """
    self.transformer = MultiTransformer(
      atom_types,
      many_body_k=many_body_k,
      max_occurs=max_occurs,
      order=order,
    )
    self.sess = tf.Session()
    self._import_model(model_path, **kwargs)

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

  def predict(self, species, coords):
    """
    Make the prediction for the given molecule.

    Args:
      species: a `List[str]` as the ordered atomic species.
      coords: a 3D array of shape `[num_examples, num_atoms, 3]` as the atomic 
        coordinates. 

    Returns:
      total_energy: a 1D array of shape `[num_examples, ]` as the predicted 
        total energies.
      atomic_energies: a 2D array of shape `[num_examples, num_atoms]` as the 
        predicted atomic energies.
      kbody_energies: a 2D array of shape `[num_examples, C(len(species), k)]` 
        as the predicted kbody energies.

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
    features, split_dims, _, weights, occurs = self.transformer.transform(
      species, coords
    )

    # Build the feed dict for running the session.
    features = features.reshape((num_mols, 1, -1, self.transformer.ck2))
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
    y_atomic = self.transformer.compute_atomic_energies(
      species, y_kbody, y_1body)

    return (np.negative(np.atleast_1d(y_total)),
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

  # Initialize a `CNNPredictor` instance. This step is relatively slow.
  tic = time.time()
  model_path = join(dirname(__file__), "models", "Bx-.v4", "model.ckpt-162026")
  calculator = CNNPredictor(["B"], model_path=model_path)
  elapsed = time.time() - tic
  print("Predictor initialized. Time: %.3f s" % elapsed)
  print("")

  print("-----------")
  print("Tests: B39-")
  print("-----------")

  coords = np.array([
    [10.9558051422, 8.4324055731, 7.1650782800],
    [12.8798039225, 9.9809890079, 10.694438864],
    [11.2518368552, 7.8085313659, 8.6417850140],
    [8.6204692072, 12.0177364175, 12.197340876],
    [7.5458126821, 11.4390544234, 9.6488879901],
    [10.8456613407, 11.6708997171, 7.1650145881],
    [8.3944716451, 7.3401128933, 9.0854876478],
    [10.3526362965, 8.1889678262, 12.785240061],
    [9.9468364743, 7.2144797836, 9.6488983306],
    [11.4976472641, 12.6466840199, 10.216873813],
    [12.0264298588, 11.5906619661, 11.310570999],
    [9.7741469986, 11.0341867464, 12.996161842],
    [8.5426200819, 12.5627665713, 10.694502532],
    [9.4952023628, 9.1164302401, 7.0045568720],
    [8.4308030550, 12.7206121860, 9.0854619208],
    [11.3580669554, 11.2702740413, 12.785123803],
    [12.1290407115, 9.1942001072, 8.0232959107],
    [13.0722871488, 9.9988610240, 9.0853531897],
    [12.4049640874, 11.4060361814, 9.6487962737],
    [10.9835146464, 10.0639706257, 7.0045352104],
    [11.2379418637, 12.2391961420, 8.6417477545],
    [10.9401722214, 9.6787761264, 12.996224479],
    [12.3688448155, 10.1860887811, 12.197317509],
    [9.9643630513, 12.7848866657, 9.7021304444],
    [9.5992709686, 12.3060475196, 8.0233165342],
    [8.1692757606, 8.5592986433, 8.0233361876],
    [10.2960239612, 7.4499879889, 11.310600580],
    [12.3612281120, 8.6385916241, 9.7020863701],
    [9.4187447099, 10.8791331889, 7.0045404488],
    [11.4749509512, 7.3798656111, 10.216879023],
    [6.9251379988, 10.0329639677, 10.216916138],
    [8.0960860914, 9.9562229122, 7.1650504080],
    [7.4078254843, 10.0118212975, 8.6417688406],
    [9.1833267140, 9.3466612012, 12.996160671],
    [8.9084241724, 7.8557575756, 12.197337874],
    [8.4752762218, 7.5157629578, 10.694522188],
    [8.1869297816, 10.6003442110, 12.785156028],
    [7.5719811474, 8.6360757658, 9.7021370730],
    [7.5753246831, 11.0188510179, 11.310557800],
  ]).reshape((1, 39, 3))
  species = list(repeat("B", 39))
  y_true = -109.6028783440 * hartree_to_ev

  y_total, y_atomic, _ = calculator.predict(species, coords)
  _print_predictions(y_total, [y_true], y_atomic, species)


if __name__ == "__main__":
  tf.app.run(main=test)
