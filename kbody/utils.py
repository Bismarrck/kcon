# coding=utf-8
"""
Some utility functions.
"""

from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

import logging
from logging.config import dictConfig

from os import getpid
from sys import platform
from subprocess import Popen, PIPE
from sys import version_info

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def lrelu(x, alpha=0.2, name=None):
  """
  A simple implementation of leaky relu.

  `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`.

  Args:
    x: a `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    alpha: a `Tensor` with type `float`, `double`.
    name: a

  Returns:
    y: a `Tensor` with the same type as `x`.

  """
  with ops.name_scope(name, "LRelu", [x]) as name:
    alpha = ops.convert_to_tensor(alpha, dtype=tf.float32, name="alpha")
    z = math_ops.multiply(alpha, x, "z")
    return math_ops.maximum(z, x, name=name)


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
