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
from collections import Counter
from functools import partial
from sklearn.metrics import pairwise_distances
from kbody_input import extract_xyz, inputs, exp_loss_weight_fn
from kbody_input import FLAGS, hartree_to_ev, au_to_angstrom
from kbody_transform import FixedLenMultiTransformer


def test_extract_mixed_xyz():
  """
  Test parsing the mixed xyz file `qm7.xyz`.
  """
  mixed_file = join(dirname(__file__), "..", "datasets", "qm7.xyz")
  array_of_species, energies, coords, _, _, _ = extract_xyz(
    mixed_file,
    num_examples=100000,
    num_atoms=23,
  )
  qm7_mat = join(dirname(__file__), "..", "datasets", "qm7.mat")
  ar = loadmat(qm7_mat)
  kcal_to_hartree = 1.0 / 627.509474
  t = ar["T"].flatten() * kcal_to_hartree * hartree_to_ev
  r = np.multiply(ar["R"], au_to_angstrom)
  lefts = np.ones(len(array_of_species), dtype=bool)
  for i in range(len(array_of_species)):
    j = np.argmin(np.abs(energies[i] - t))
    if lefts[j]:
      n = len(array_of_species[i])
      d = np.linalg.norm(coords[i][:n] - r[j][:n])
      if d < 0.1:
        lefts[j] = False
  assert not np.all(lefts)


def test_build_dataset():
  """
  Test building the mixed QM7 dataset.
  """
  dataset = 'qm7.test'
  xyzfile = join("..", "datasets", "qm7.xyz")
  array_of_species, energies, coordinates, _, _, _ = extract_xyz(
    xyzfile,
    num_examples=7165,
    num_atoms=23,
    verbose=False,
  )
  indices = list(range(len(coordinates)))

  r_hh = 0.64
  many_body_k = 3
  max_occurs = {}
  for symbols in array_of_species:
    c = Counter(symbols)
    for specie, times in c.items():
      max_occurs[specie] = max(max_occurs.get(specie, 0), times)

  clf = FixedLenMultiTransformer(
    max_occurs, many_body_k=many_body_k, periodic=False
  )

  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    array_of_species,
    energies,
    coordinates,
    train_file,
    indices=indices,
  )

  with tf.Session() as sess:

    batch = inputs(train=True, shuffle=False, batch_size=5, dataset=dataset)
    tf.train.start_queue_runners(sess=sess)

    # --------
    # Features
    # --------
    features, weights = sess.run([batch[0], batch[3]])
    species = array_of_species[1]
    n = len(species)
    coords = coordinates[1][:n]
    dists = pairwise_distances(coords[2:5])
    v = np.sort(np.exp(-dists / r_hh)[[0, 0, 1], [1, 2, 2]])

    # 2289 is the starting index of `HHH`.
    assert np.linalg.norm(v - features[1, 0, 2289]) < 0.001
    assert np.abs(weights[1].flatten().sum() - comb(n, 3)) < 0.001

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

  array_of_species, energies, coordinates, _, _, _ = extract_xyz(
    xyzfile,
    xyz_format='grendel',
    num_examples=5000,
    num_atoms=17,
    verbose=False,
  )
  indices = list(range(len(coordinates)))
  min_ener = energies.min()

  many_body_k = 3
  max_occurs = {}
  for symbols in array_of_species:
    c = Counter(symbols)
    for specie, times in c.items():
      max_occurs[specie] = max(max_occurs.get(specie, 0), times)

  clf = FixedLenMultiTransformer(
    max_occurs, many_body_k=many_body_k, periodic=False
  )

  beta = 1.0 / 10.0
  loss_weight_fn = partial(exp_loss_weight_fn, x0=min_ener, beta=beta)

  train_file = join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset))
  clf.transform_and_save(
    array_of_species,
    energies,
    coordinates,
    train_file,
    indices=indices,
    loss_weight_fn=loss_weight_fn
  )

  with tf.Session() as sess:

    batch = inputs(train=True, shuffle=False, batch_size=5, dataset=dataset)
    tf.train.start_queue_runners(sess=sess)

    loss_weights = sess.run(batch[4])
    for i in range(5):
      assert np.abs(loss_weight_fn(energies[i]) - loss_weights[i]) < 0.000001

    time.sleep(5)

  # Delete this dataset file.
  remove(join(FLAGS.binary_dir, "{}-train.json".format(dataset)))
  remove(join(FLAGS.binary_dir, "{}-train.tfrecords".format(dataset)))


if __name__ == '__main__':
  unittest.main()
