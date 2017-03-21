# coding=utf-8
"""
Some utility functions.
"""

from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def inverse_decay(init_learning_rate, epoch, decay_factor, name=None):
  """
  The inverse decay function.

  Args:
    init_learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_epoch: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global epoch to use for the decay computation.  Must not be negative.
    decay_factor: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay equation above.
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  """
  if epoch is None:
    raise ValueError("global_step is required for inv_decay.")
  with ops.name_scope(name, "InvDecay",
                      [init_learning_rate, epoch, decay_factor]) as name:
    init_learning_rate = ops.convert_to_tensor(
      init_learning_rate, name="init_learning_rate")
    dtype = init_learning_rate.dtype
    epoch = math_ops.cast(epoch, dtype)
    decay_factor = math_ops.cast(decay_factor, dtype)
    top = math_ops.multiply(init_learning_rate, decay_factor)
    return math_ops.div(top, math_ops.add(epoch, decay_factor), name=name)


def tanh_increase(init_learning_rate, global_step, decay_step, decay_factor,
                  staircase=True, name=None):
  """
  Gradually increase the initial learning rate with the hyperbolic tangent
  function.

  Args:
    init_learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number. The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global epoch to use for the decay computation.  Must not be negative.
    decay_step: A scalar `int32` or `int64` `Tensor` or a Python number.
    decay_factor: decay_factor: A scalar `int32` or `int64` `Tensor` or a Python
      number. Must be positive.  See the decay equation above.
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`. The gradually
    increased learning rate.

  """
  with ops.name_scope(
      name,
      "TanhDecay",
      [init_learning_rate, global_step, decay_step, decay_factor]) as name:
    init_learning_rate = ops.convert_to_tensor(
      init_learning_rate, name="init_learning_rate")
    decay_factor = math_ops.multiply(2.0, decay_factor)
    global_step = tf.cast(global_step, tf.float32)
    decay_step = tf.cast(decay_step, tf.float32)
    alpha = math_ops.div(global_step, decay_step)
    if staircase:
      alpha = math_ops.floor(alpha)
    z = math_ops.multiply(alpha, decay_factor)
    z = math_ops.exp(z)
    z = math_ops.div(math_ops.subtract(z, 1.0), math_ops.add(z, 1.0))
    return math_ops.add(z, init_learning_rate, name=name)


def test_tanh_increase():

  global_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)
  inc_op = tf.assign(global_step, tf.add(global_step, 1.0))
  tl = tanh_increase(0.01, global_step, 10, 0.005)

  with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in range(100):
      rate = sess.run(tl)
      print("step: %3d, rate: %f" % (i, rate))
      sess.run(inc_op)


if __name__ == "__main__":
  test_tanh_increase()
