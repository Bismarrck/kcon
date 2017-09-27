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
from os.path import join
from sys import platform
from subprocess import Popen, PIPE
from sys import version_info

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


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


def set_logging_configs(debug=False, logfile="logfile"):
  """
  Set 
  """
  if debug:
    level = logging.DEBUG
    handlers = ['console', 'file']
  else:
    level = logging.INFO
    handlers = ['file']

  LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
      # For files
      'detailed': {
        'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
      },
      # For the console
      'console': {
        'format': '[%(levelname)s] %(message)s'
      }
    },
    "handlers": {
      'console': {
        'class': 'logging.StreamHandler',
        'level': logging.DEBUG,
        'formatter': 'console',
      },
      'file': {
        'class': 'logging.FileHandler',
        'level': logging.INFO,
        'formatter': 'detailed',
        'filename': logfile,
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
