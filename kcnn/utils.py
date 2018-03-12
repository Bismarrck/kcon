# coding=utf-8
"""
Some utility functions.
"""

from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import logging
import json
import time
from scipy.misc import factorial, comb
from logging.config import dictConfig
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.layers import variance_scaling_initializer
from os import getpid
from os.path import join, dirname
from sys import platform
from subprocess import Popen, PIPE
from sys import version_info

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Gauss:
  """
  A simple implementation of the Gaussian function.
  """

  def __init__(self, mu, sigma):
    """
    Initialization method.

    Args:
      mu: a `float` as the center of the gaussian.
      sigma: a `float` as the sigma.

    """
    self.mu = mu
    self.sigma = sigma
    self.beta = 1.0 / (2.0 * sigma**2)
    self.c = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

  def __call__(self, x):
    """
    Compute the values at `x`.
    """
    return self.c * np.exp(-self.beta * (x - self.mu)**2)


def compute_n_from_cnk(cnk, k):
  """
  Return the correponding N given C(N, k) and k.

  Args:
    cnk: an `int` as the value of C(N, k).
    k: an `int` as the value of k.

  Returns:
    n: an `int` as the value of N.

  """
  if k == 2:
    return int((1 + np.sqrt(1 + 8 * cnk)) * 0.5)
  else:
    istart = int(np.floor(np.power(factorial(k) * cnk, 1.0 / k)))
    for v in range(istart, istart + k):
      if comb(v, k) == cnk:
        return v
    else:
      raise ValueError(
        "The N for C(N, {}) = {} cannot be solved!".format(k, cnk))


def safe_divide(a, b):
  """
  Safe division while ignoring / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0].

  References:
     https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero

  """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide(a, b)
    c[~ np.isfinite(c)] = 0  # -inf inf NaN
  return c


def get_atoms_from_kbody_term(kbody_term):
  """
  Return the atoms in the given k-body term.

  Args:
    kbody_term: a `str` as the k-body term.

  Returns:
    atoms: a `list` of `str` as the chemical symbols of the atoms.

  """
  sel = [0]
  for i in range(len(kbody_term)):
    if kbody_term[i].isupper():
      sel.append(i + 1)
    else:
      sel[-1] += 1
  atoms = []
  for i in range(len(sel) - 1):
    atoms.append(kbody_term[sel[i]: sel[i + 1]])
  return atoms


def get_k_from_var(var):
  """
  Get the associated `k` for the given variable.

  Args:
    var: a `tf.Variable`.

  Returns:
    k: an `int`.

  """
  name = var.op.name
  elements = name.split("/")
  if len(elements) >= 4 and elements[0] == 'kCON':
    symbols = get_atoms_from_kbody_term(elements[1])
    return len(symbols) - symbols.count("X")
  else:
    return -1


def lrelu(x, alpha=0.01, name=None):
  """
  The leaky relu activation function.

  `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`.

  Args:
    x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    alpha: a `float32` tensor as the alpha.
    name: a `str` as the name of this op.

  Returns:
    y: a `Tensor` with the same type as `x`.

  """
  with ops.name_scope(name, "LRelu", [x]) as name:
    alpha = ops.convert_to_tensor(alpha, dtype=tf.float32, name="alpha")
    z = math_ops.multiply(alpha, x, "z")
    return math_ops.maximum(z, x, name=name)


def selu(x, name=None):
  """
  The Scaled Exponential Linear Units.

  Args:
    x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: a `str` as the name of this op.

  Returns:
    y: a `Tensor` with the same type as `x`.

  References:
    https://arxiv.org/pdf/1706.02515.pdf

  """
  with ops.name_scope(name, "SeLU", [x]):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    elu = tf.nn.elu(x, name='elu')
    z = tf.multiply(elu, alpha, name='z')
    return scale * tf.where(x >= 0.0, x, z, name='selu')


def selu_initializer(dtype=tf.float32, seed=None):
  """
  The weights initializer for selu activations.

  Args:
    seed: A Python integer. Used to create random seeds. See
          @{tf.set_random_seed} for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a weight matrix.

  """
  mode = 'FAN_IN'
  return variance_scaling_initializer(
    factor=1.0, mode=mode, dtype=dtype, seed=seed)


def reduce_l2_norm(tensor, name=None):
  """
  Return the mean of the L2 norms along axis 1 of the given tensor.

  Args:
    tensor: a `float32` tensor of rank 2.
    name: a `str` as the name of the op.

  Returns:
    norm: a `float32` tensor as the mean of the L2 norms.

  """
  with ops.name_scope(name, "reduce_norm", [tensor]):
    norms = tf.norm(tensor, axis=1, keep_dims=False, name="norm")
    return tf.reduce_mean(norms, name="mean")


def msra_initializer(dtype=tf.float32, seed=None):
  """
  [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)

  Args:
    seed: A Python integer. Used to create random seeds. See
          @{tf.set_random_seed} for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer for a weight matrix.

  """
  return variance_scaling_initializer(seed=seed, dtype=dtype)


def get_xargs(pid=None):
  """
  Return the build and execute command lines from standard input for a process. 
  This is a wrapper of the linux command `xargs -0 <`. maxOS and Win 
  are not supported.
  
  Args:
    pid: an `int` as the target process. 
  
  Returns:
    exe_args: a `str` as the execute command line for process `pid` or None.

  """
  if platform != "linux":
    return None
  if not pid:
    pid = getpid()

  p = Popen("xargs -0 < /proc/{}/cmdline".format(pid), stdout=PIPE, stderr=PIPE,
            shell=True)
  stdout, stderr = p.communicate()
  if len(stderr) > 0:
    return None
  elif version_info > (3, 0):
    return bytes(stdout).decode()
  else:
    return stdout


def set_logging_configs(debug=False, logfile="logfile", is_eval=False):
  """
  Config the logging module.
  """
  if is_eval:
    level = logging.INFO
    handlers = ['normal', 'eval']
  elif debug:
    level = logging.DEBUG
    handlers = ['console', 'normal']
  else:
    level = logging.INFO
    handlers = ['normal']

  LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
      # For normal logs
      'detailed': {
        'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
      },
      # For the console
      'console': {
        'format': '[%(levelname)s] %(message)s'
      },
      # For the evaluations results
      'simple': {
        'format': "%(message)s"
      }
    },
    "handlers": {
      'console': {
        'class': 'logging.StreamHandler',
        'level': logging.DEBUG,
        'formatter': 'console',
      },
      # Redirect all logs to the file `logfile`.
      'normal': {
        'class': 'logging.FileHandler',
        'level': logging.INFO,
        'formatter': 'detailed',
        'filename': logfile,
        'mode': 'a',
      },
      # Redirect the simplfied evaluation results to the file SUMMARY.
      'eval': {
        'class': 'logging.FileHandler',
        'level': logging.CRITICAL,
        'formatter': 'simple',
        'filename': join(dirname(logfile), 'SUMMARY'),
        'mode': 'a',
      }
    },
    "root": {
      'handlers': handlers,
      'level': level,
    },
    "disable_existing_loggers": False
  }
  dictConfig(LOGGING_CONFIG)


def save_training_flags(save_dir, args):
  """
  Save the training flags to the train_dir.
  """
  args["run_flags"] = " ".join(
    ["--{}={}".format(k, v) for k, v in args.items()]
  )
  cmdline = get_xargs()
  if cmdline:
    args["cmdline"] = cmdline
  timestamp = "{}".format(int(time.time()))
  with open(join(save_dir, "flags.{}.json".format(timestamp)), "w+") as f:
    json.dump(args, f, indent=2)
