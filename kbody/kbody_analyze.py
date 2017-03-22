# coding=-utf8
"""
Do post-analysis.
"""
from __future__ import absolute_import, print_function, division

import math
import time
import numpy as np
import tensorflow as tf
import kbody
from scipy.misc import comb

__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './events/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './events',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_evals', 500,
                            """Number of examples to run.""")


def analyze_once(saver, y_true_op, y_pred_op, kbody_contrib_op, cnks,
                 test_indices):
  """
  Run Eval once.

  Args:
    saver: Saver.
    y_true_op: the Tensor for fetching real energies.
    y_pred_op: the Tensor for fetching predicted energies.
    kbody_contrib_op: the Tensor for getting predicted kbody contributions.
    cnks: an array of shape [C(N,k), C(k,2)] as the indices of the atoms of all
      kbody terms.
    test_indices: the indices of the testing examples in the original data file.

  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_evals / FLAGS.batch_size))
      cnk = int(comb(FLAGS.num_atoms, FLAGS.many_body_k, exact=True))
      ck2 = int(comb(FLAGS.many_body_k, 2, exact=True))
      y_true = np.zeros((FLAGS.num_evals, ), dtype=np.float32)
      y_pred = np.zeros((FLAGS.num_evals, ), dtype=np.float32)
      kbody_contribs = np.zeros((FLAGS.num_evals, cnk), dtype=np.float32)
      step = 0

      while step < num_iter and not coord.should_stop():

        y_true_, y_pred_, kbody_contribs_ = sess.run(
          [y_true_op, y_pred_op, kbody_contrib_op]
        )
        maes[step] = mae_val
        istart = step * FLAGS.batch_size
        istop = min(istart + FLAGS.batch_size, FLAGS.num_evals)
        y_true[istart: istop] = y_true_
        y_pred[istart: istop] = y_pred_
        kbody_contribs[istart: istop, :] = kbody_contribs_
        step += 1

      atomic_energies = np.zeros((FLAGS.num_evals, FLAGS.num_atoms),
                                 dtype=np.float32)

      for step in range(FLAGS.num_evals):
        for i in range(cnk):
          for j in cnks[i]:
            atomic_energies[step, j] -= kbody_contribs[step][i] / ck2

      np.savez("atomics.npz",
               atomic_energies=atomic_energies,
               indices=test_indices)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:

    # Read dataset configurations
    settings = kbody.inputs_settings(train=False)
    offsets = settings["kbody_term_sizes"]
    selections = settings["kbody_selections"]
    indices = settings["inverse_indices"]
    kbody_terms = [x.replace(",", "") for x in settings["kbody_terms"]]
    cnks = []
    for kbody_term in settings["kbody_terms"]:
      kbody_selections = selections[kbody_term]
      cnks.extend(kbody_selections)
    cnks = np.array(cnks)

    # Get features and energies for evaluation.
    features, y_true = kbody.inputs(train=False, shuffle=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    y_pred, kbody_contribs = kbody.inference(
      features,
      offsets,
      kbody_terms=kbody_terms,
      verbose=True,
    )
    y_true = tf.cast(y_true, tf.float32)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        kbody.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Evaluate
    while True:
      analyze_once(saver, y_true, y_pred, kbody_contribs, cnks, indices)
      time.sleep(5)
      break


# pylint: disable=unused-argument
def main(argv=None):
  evaluate()


if __name__ == '__main__':
  tf.app.run()
