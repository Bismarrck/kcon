# coding=utf-8
"""
This module offers auxiliary functions for add summaries.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def add_total_norm_summaries(grads_and_vars, collection,
                             only_summary_total=True):
  """
  Add summaries for the 2-norms of the gradients.

  Args:
    grads_and_vars: a list of (gradient, variable) returned by an optimizer.
    collection: a `str` as the collection to add the computed norms.
    only_summary_total: a `bool` indicating whether we should summarize the
      individual norms or not.

  Returns:
    total_norm: a `float32` tensor that computes the sum of all norms of the
      gradients.

  """
  for grad, var in grads_and_vars:
    if grad is not None:
      norm = tf.norm(grad, name=var.op.name + "/norm")
      tf.add_to_collection(collection, norm)
      with tf.name_scope("gradients/{}/".format(collection)):
        tf.summary.histogram(var.op.name + "/hist", grad)
      if not only_summary_total:
        with tf.name_scope("gradients/{}/".format(collection)):
          tf.summary.scalar(var.op.name + "/norm", norm)

  with tf.name_scope("total_norm/"):
    total_norm = tf.add_n(tf.get_collection(collection))
    tf.summary.scalar(collection, total_norm)

  return total_norm


def add_variable_summaries():
  """
  Add variable summaries.
  """
  with tf.name_scope("variables"):

    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
      vsum = tf.reduce_sum(tf.abs(var, name="absolute"), name="vsum")

      if not var.op.name.startswith('kCON/one-body'):
        tf.add_to_collection('k_sum', vsum)
      else:
        tf.add_to_collection('1_sum', vsum)

    tf.summary.scalar(
      'kbody', tf.add_n(tf.get_collection('k_sum'), name='vars_ksum'))
    tf.summary.scalar(
      '1body', tf.add_n(tf.get_collection('1_sum'), name='vars_1sum'))
