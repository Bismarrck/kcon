# coding=utf-8
"""
This module is used for making predictions with trained models.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import kbody_transform
import time
from os.path import join, dirname

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_tests", 1000,
                            """The number of tests to run.""")


class CNNPredictor:
  """
  An energy predictor based on the deep neural network of 'sum-kbody-cnn'.
  """

  def __init__(self, atom_types, model_path, many_body_k=3, max_occurs=None,
               **kwargs):
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
    self.transformer = kbody_transform.MultiTransformer(
      atom_types,
      many_body_k=many_body_k,
      max_occurs=max_occurs,
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
    features = features.reshape((num_mols, 1, -1, self.transformer.ck2))
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
    y_total, y_kbody = self.sess.run(
      [self.y_total_op, self.y_kbody_op],
      feed_dict=feed_dict
    )

    # Transform the kbody energies to atomic energies.
    y_kbody = np.squeeze(y_kbody, axis=(1, 3))
    y_atomic = self.transformer.compute_atomic_energies(species, y_kbody)

    return np.negative(y_total), np.negative(y_atomic)


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


def _test(calculator):
  """
  Run unittests for `CNNPredictor`. The dataset for these tests is `C9H7Nv1`.
  """

  import kbody_input

  cwd = dirname(__file__)
  xyz_file = join(cwd, "..", "datasets", "C9H7Nv1.xyz")
  num_examples = 5000
  num_atoms = 17
  num_tests = num_examples

  # Extract structures from the file.
  species, energies, coords, _ = kbody_input.extract_xyz(
    xyz_file,
    num_examples,
    num_atoms,
    xyz_format='grendel',
    verbose=False
  )

  # Make final predictions.
  tic = time.time()
  slicer = np.random.choice(range(num_examples), num_tests)
  y_total, y_atomic = calculator.predict(species, coords[slicer])
  elapsed = time.time() - tic
  speed = float(num_tests) / elapsed
  print("Prediction time: %.3f s, speed: %.2f examples/s" % (elapsed, speed))
  print("")

  _print_predictions(y_total, energies, y_atomic, species)


def _test_small(calculator):
  """
  Test the `CNNPredictor` of CxHyN with a CH4 molecule.
  """
  species = ["C", "H", "H", "H", "H"]
  coords = np.array([
    [0.15625000,    1.42857141,    0.00000000],
    [0.51290443,    0.41976140,    0.00000000],
    [0.51292284,    1.93296960,    0.87365150],
    [0.51292284,    1.93296960,   -0.87365150],
    [-0.91375000,   1.42858459,    0.00000000]
  ], dtype=np.float64).reshape((1, 5, 3))

  y_total, y_atomic = calculator.predict(species, coords)
  _print_predictions(y_total, np.zeros_like(y_total), y_atomic, species)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def test(unused):

  cwd = dirname(__file__)
  model_name = "model.ckpt-1077218"
  model_path = join(cwd, "models", "C9H7N.v2", model_name)

  # Initialize a `CNNPredictor` instance. This step is relatively slow.
  tic = time.time()
  calculator = CNNPredictor(
    ["C", "H", "N"],
    model_path=model_path,
    max_occurs={"N": 1}
  )
  elapsed = time.time() - tic
  print("Predictor initialized. Time: %.3f s" % elapsed)
  print("")

  for unittest, name in [(_test, "C9H7N"), (_test_small, "Small Molecules")]:
    print("------")
    print("Tests:", name)
    print("------")
    unittest(calculator)
    print("")


if __name__ == "__main__":
  tf.app.run(main=test)
