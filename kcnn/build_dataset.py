#!coding=utf-8
"""
This module is used to build datasets. The dataset xyz files must be saved in
dir '../datasets'.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import transformer
from functools import partial
from os.path import join, isfile
from database import Database
from pipeline import get_filenames

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


tf.app.flags.DEFINE_float('unit', None,
                          """Override the default unit if this is not None.""")
tf.app.flags.DEFINE_boolean('periodic', False,
                            """The isomers are periodic structures.""")
tf.app.flags.DEFINE_string('format', 'xyz',
                           """The format of the xyz file.""")
tf.app.flags.DEFINE_integer('num_examples', 5000,
                            """The total number of examples to use.""")
tf.app.flags.DEFINE_float('test_size', 0.2,
                          """The proportion of the dataset to include in the 
                          test split""")
tf.app.flags.DEFINE_integer("norm_order", 1,
                            """The exponential order for normalizing 
                            distances.""")
tf.app.flags.DEFINE_float('weighted_loss', None,
                          """The kT (eV) for computing the weighted loss. """)

FLAGS = tf.app.flags.FLAGS


def exponentially_weighted_loss(x, x0=0.0, beta=1.0):
  """
  An exponential function for computing the weighted loss.

  I.e. \\(y = \e^{-\beta \cdot (x - x_0)}\\).
  """
  return np.float32(np.exp(-(x - x0) * beta))


def may_build_dataset(dataset=None, verbose=True):
  """
  Build the dataset if needed.

  Args:
    verbose: boolean indicating whether the building progress shall be printed
      or not.
    dataset: a `str` as the name of the dataset.

  """
  train_file, _ = get_filenames(train=True, dataset_name=dataset)
  test_file, _ = get_filenames(train=False, dataset_name=dataset)

  # Check if the xyz file is accessible.
  xyzfile = join("..", "datasets", "{}.xyz".format(FLAGS.dataset))
  if not isfile(xyzfile):
    raise IOError("The dataset file %s can not be accessed!" % xyzfile)

  # Extract the xyz file and split it into two sets: a training set and a
  # testing set.
  if FLAGS.forces and FLAGS.format != 'ase':
    raise ValueError("Currently only ASE-generated xyz files are supported if "
                     "forces training is enabled.")

  database = Database.from_xyz(xyzfile,
                               num_examples=FLAGS.num_examples,
                               verbose=verbose,
                               xyz_format=FLAGS.format,
                               unit_to_ev=FLAGS.unit)
  database.split(test_size=min(max(FLAGS.test_size, 0.0), 1.0))

  # The maximum supported `k` is 5.
  k_max = min(5, FLAGS.k_max)

  # Determine the maximum occurances of each atom type.
  max_occurs = database.max_occurs

  # Setup the exponential-scaled RMSE.
  if FLAGS.weighted_loss is None:
    exp_rmse_fn = None
  else:
    min_ener, _ = database.energy_range
    beta = 1.0 / FLAGS.weighted_loss
    exp_rmse_fn = partial(exponentially_weighted_loss, x0=min_ener, beta=beta)

  # Use a `FixedLenMultiTransformer` to generate features because it will be
  # much easier if the all input samples are fixed-length.
  clf = transformer.FixedLenMultiTransformer(
    max_occurs=max_occurs,
    k_max=k_max,
    periodic=FLAGS.periodic,
    norm_order=FLAGS.norm_order,
    include_all_k=FLAGS.include_all_k,
    atomic_forces=FLAGS.forces
  )
  clf.transform_and_save(
    database,
    train_file=train_file,
    test_file=test_file,
    verbose=True,
    loss_fn=exp_rmse_fn
  )


def main(_):
  """
  The main function.
  """
  if FLAGS.periodic and (FLAGS.format != 'ase'):
    tf.logging.error(
      "The xyz format must be `grendel` if `periodic` is True!")
  else:
    may_build_dataset(verbose=True)


if __name__ == "__main__":
  tf.app.run(main=main)
