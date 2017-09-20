#!coding=utf-8
"""
Unittests of the module `kbody_input`.
"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import unittest
import time
from scipy.io import loadmat
from scipy.misc import comb
from os.path import join, isfile, dirname
from os import remove
from functools import partial
from sklearn.metrics import pairwise_distances
from reader import y_inputs
from build_dataset import exponentially_weighted_loss
from reader import FLAGS
from database import Database
from constants import hartree_to_ev, au_to_angstrom
from transformer import FixedLenMultiTransformer


def test_extract_mixed_xyz():
  """
  Test parsing the mixed xyz file `qm7.xyz`.
  """
  mixed_file = join(dirname(__file__), "..", "datasets", "qm7.xyz")
  database = Database.from_xyz(mixed_file, num_examples=100000)
  qm7_mat = join(dirname(__file__), "..", "datasets", "qm7.mat")
  ar = loadmat(qm7_mat)
  kcal_to_hartree = 1.0 / 627.509474
  t = ar["T"].flatten() * kcal_to_hartree * hartree_to_ev
  r = np.multiply(ar["R"], au_to_angstrom)
  lefts = np.ones(database.num_examples, dtype=bool)
  for i in range(database.num_examples):
    atoms = database[i]
    j = np.argmin(np.abs(atoms.get_total_energy() - t))
    if lefts[j]:
      n = len(atoms)
      coords = atoms.get_positions()
      d = np.linalg.norm(coords[:n] - r[j][:n])
      if d < 0.1:
        lefts[j] = False
  assert not np.all(lefts)


def test_build_dataset():
  """
  Test building the mixed QM7 dataset.
  """
  radius = 1.5
  dataset = 'qm7.test'
  xyzfile = join("..", "datasets", "qm7.xyz")
  database = Database.from_xyz(xyzfile, num_examples=7165, verbose=False)
  k_max = 3
  clf = FixedLenMultiTransformer(
    database.max_occurs, k_max=k_max, periodic=False
  )
  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    database, train_file=train_file, verbose=True
  )

  atoms = database[database.ids_of_training_examples[1]]
  species = atoms.get_chemical_symbols()
  coords = atoms.get_positions()

  with tf.Session() as sess:

    batch = y_inputs(
      train=True,
      shuffle=False,
      batch_size=5,
      dataset_name=dataset
    )
    tf.train.start_queue_runners(sess=sess)

    # --------
    # Features
    # --------
    features, weights = sess.run([batch[0], batch[3]])
    natoms = len(species)
    coords = coords[:natoms]
    dists = pairwise_distances(coords[:3])
    v = np.sort(np.exp(-dists / radius)[[0, 0, 1], [1, 2, 2]])

    # 0 is the starting index of `CCC`.
    assert np.linalg.norm(v - features[1, 0, 0]) < 0.001
    assert np.abs(weights[1].flatten().sum() - comb(natoms, 3)) < 0.001

    time.sleep(5)

  # Delete this dataset file.
  remove(join(FLAGS.binary_dir, "{}-train.json".format(dataset)))
  remove(join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset)))


def test_compute_loss_weight():
  """
  Test the function for computing loss weights using the dataset of `C9H7N.PBE`.
  """
  dataset = 'C9H7N.PBE.test'

  xyzfile = join("..", "datasets", "C9H7N.PBE.xyz")
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  database = Database.from_xyz(xyzfile, xyz_format='grendel', num_examples=5000,
                               verbose=False)
  min_ener, _ = database.energy_range
  k_max = 3

  clf = FixedLenMultiTransformer(
    database.max_occurs, k_max=k_max, periodic=False
  )

  beta = 1.0 / 10.0
  exp_loss_fn = partial(exponentially_weighted_loss, x0=min_ener, beta=beta)

  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    database,
    train_file=train_file,
    loss_fn=exp_loss_fn
  )

  objects = database[database.ids_of_training_examples[:5]]
  y_true = [atoms.get_total_energy() for atoms in objects]

  with tf.Session() as sess:
    batch = y_inputs(
      train=True,
      shuffle=False,
      batch_size=5,
      dataset_name=dataset
    )
    tf.train.start_queue_runners(sess=sess)
    y_weights = sess.run(batch[4])
    for i in range(5):
      assert np.abs(exp_loss_fn(y_true[i]) - y_weights[i]) < 0.000001

    time.sleep(5)

  # Delete this dataset file.
  remove(join(FLAGS.binary_dir, "{}-train.json".format(dataset)))
  remove(join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset)))


if __name__ == '__main__':
  unittest.main()
