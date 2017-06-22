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
from kbody_input import inputs, exp_rmse_loss_fn
from xyz import extract_xyz
from kbody_input import FLAGS
from constants import hartree_to_ev, au_to_angstrom
from kbody_transform import FixedLenMultiTransformer


def test_extract_mixed_xyz():
  """
  Test parsing the mixed xyz file `qm7.xyz`.
  """
  mixed_file = join(dirname(__file__), "..", "datasets", "qm7.xyz")
  xyz = extract_xyz(mixed_file, num_examples=100000, num_atoms=23)
  qm7_mat = join(dirname(__file__), "..", "datasets", "qm7.mat")
  ar = loadmat(qm7_mat)
  kcal_to_hartree = 1.0 / 627.509474
  t = ar["T"].flatten() * kcal_to_hartree * hartree_to_ev
  r = np.multiply(ar["R"], au_to_angstrom)
  lefts = np.ones(xyz.num_examples, dtype=bool)
  for i in range(xyz.num_examples):
    species, energy, coords = xyz[i][:3]
    j = np.argmin(np.abs(energy - t))
    if lefts[j]:
      n = len(species)
      d = np.linalg.norm(coords[:n] - r[j][:n])
      if d < 0.1:
        lefts[j] = False
  assert not np.all(lefts)


def test_build_dataset():
  """
  Test building the mixed QM7 dataset.
  """
  l = 1.5
  dataset = 'qm7.test'
  xyzfile = join("..", "datasets", "qm7.xyz")
  xyz = extract_xyz(xyzfile, num_examples=7165, num_atoms=23, verbose=False)
  many_body_k = 3
  clf = FixedLenMultiTransformer(
    xyz.get_max_occurs(), many_body_k=many_body_k, periodic=False
  )
  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    xyz, train_file=train_file, verbose=True
  )

  species, y_true, coords, = xyz[xyz.indices_of_training[1]][:3]

  with tf.Session() as sess:

    batch = inputs(train=True, shuffle=False, batch_size=5, dataset=dataset)
    tf.train.start_queue_runners(sess=sess)

    # --------
    # Features
    # --------
    features, weights = sess.run([batch[0], batch[3]])
    natoms = len(species)
    coords = coords[:natoms]
    dists = pairwise_distances(coords[:3])
    v = np.sort(np.exp(-dists / l)[[0, 0, 1], [1, 2, 2]])

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

  xyz = extract_xyz(xyzfile, xyz_format='grendel', num_examples=5000,
                    num_atoms=17, verbose=False)
  min_ener, _ = xyz.energy_range

  many_body_k = 3

  clf = FixedLenMultiTransformer(
    xyz.get_max_occurs(), many_body_k=many_body_k, periodic=False
  )

  beta = 1.0 / 10.0
  exp_loss_fn = partial(exp_rmse_loss_fn, x0=min_ener, beta=beta)

  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    xyz,
    train_file=train_file,
    exp_rmse_fn=exp_loss_fn
  )
  y_true = xyz.get_training_samples()[1]

  with tf.Session() as sess:
    batch = inputs(train=True, shuffle=False, batch_size=5, dataset=dataset)
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
