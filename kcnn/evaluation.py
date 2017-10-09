# coding=-utf8
"""
The evaluation module for KCNN.
"""
from __future__ import absolute_import, division, print_function

import math
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from kcnn import kcnn_from_dataset
from constants import VARIABLE_MOVING_AVERAGE_DECAY
from utils import set_logging_configs
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from os.path import join

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('checkpoint_dir', './events',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('summary_dir', None,
                           """Alternative dir for saving summaries.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 300,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_evals', 500,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('logfile', 'eval.log',
                           """The file to write evaluation logs.""")
tf.app.flags.DEFINE_boolean('output_acc_error', False,
                            """Output the accumulative error.""")
tf.app.flags.DEFINE_integer('stop_after_repeats', 3,
                            """Automatically stop the evaluation if a checkpoint 
                            was repeatedly used N times.""")


def get_eval_dir():
  """
  Return the evaluation dir.
  """
  return join(FLAGS.checkpoint_dir, "eval")


def eval_once(saver, summary_writer, y_true_op, y_nn_op, f_true_op, f_nn_op,
              summary_op):
  """
  Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    y_true_op: The Tensor used for fetching true predictions.
    y_nn_op: The Tensor used for fetching neural network predicted energies.
    f_true_op: The Tensor used for fetching true forces.
    f_nn_op: The Tensor used for fetching neural network predicted forces.
    summary_op: Summary op.

  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      global_step = int(global_step)
    else:
      tf.logging.info('No checkpoint file found')
      return -1

    # Wait until the parsed global step is not zero.
    if (not FLAGS.run_once) and global_step <= 1:
      tf.logging.info("The global step is <= 1. Wait ...")
      return global_step

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []

    try:
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
          sess,
          coord=coord,
          daemon=True,
          start=True)
        )

      num_evals = FLAGS.num_evals
      num_iter = int(math.ceil(num_evals / FLAGS.batch_size))
      atomic_forces = FLAGS.forces

      y_true = np.zeros((num_evals, ), dtype=np.float32)
      y_pred = np.zeros((num_evals, ), dtype=np.float32)

      if atomic_forces:
        num_entries = f_true_op.get_shape().as_list()[1]
        f_true = np.zeros((num_evals, num_entries), dtype=np.float32)
        f_pred = np.zeros((num_evals, num_entries), dtype=np.float32)
      else:
        num_entries = 0
        f_true = None
        f_pred = None

      step = 0
      tic = time.time()

      while step < num_iter and not coord.should_stop():

        istart = step * FLAGS.batch_size
        istop = min(istart + FLAGS.batch_size, num_evals)

        y_true_, y_pred_ = sess.run([y_true_op, y_nn_op])
        y_true[istart: istop] = -y_true_
        y_pred[istart: istop] = -y_pred_

        if atomic_forces:
          f_true_, f_pred_ = sess.run([f_true_op, f_nn_op])
          f_true[istart: istop, :] = f_true_
          f_pred[istart: istop, :] = f_pred_

        step += 1

      elpased = time.time() - tic

      # Compute the common evaluation metrics.
      precision = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      y_diff = np.abs(y_true - y_pred)
      emax = y_diff.max()
      emin = y_diff.min()
      score = r2_score(y_true, y_pred)

      dtime = datetime.now()
      tf.logging.info("%s: step      = %d" % (dtime, global_step))
      tf.logging.info("%s: time      = %.2f" % (dtime, elpased))
      tf.logging.info("%s: precision = %10.6f" % (dtime, precision))
      tf.logging.info("%s: RMSE      = %10.6f" % (dtime, rmse))
      tf.logging.info("%s: minimum   = %10.6f" % (dtime, emin))
      tf.logging.info("%s: maximum   = %10.6f" % (dtime, emax))
      tf.logging.info("%s: score     = %10.6f" % (dtime, score))

      if atomic_forces:
        f_precision = mean_absolute_error(f_true, f_pred)
        f_rmse = np.sqrt(mean_squared_error(f_true, f_pred))
        f_true_norms = np.linalg.norm(f_true, axis=1)
        f_diff_norms = np.linalg.norm(f_true - f_pred, axis=1)
        f_ratio = (f_diff_norms / f_true_norms).mean()
        tf.logging.info("%s: f_MAE     = %10.6f" % (dtime, f_precision))
        tf.logging.info("%s: f_RMSE    = %10.6f" % (dtime, f_rmse))
        tf.logging.info("%s: f_ratio   = %10.6f" % (dtime, f_ratio))

      # Randomly output 10 predictions and true values
      if FLAGS.output_acc_error:
        y_diff = y_diff[np.argsort(y_true)]
        for i in range(1, 10):
          percent = float(i) / 10.0
          n = int(percent * num_evals)
          mae = y_diff[:n].mean()
          tf.logging.info(" * %2d%% MAE: % 8.3f" % (percent * 100.0, mae))

      else:
        indices = np.random.choice(range(FLAGS.num_evals), size=10)
        for i in indices:
          tf.logging.info(
            " * Predicted: % 15.6f,  Real: % 15.6f" % (y_pred[i], y_true[i]))
        if atomic_forces:
          index = np.random.randint(low=0, high=FLAGS.num_evals)
          f_pred_vec = np.reshape(f_pred[index], (-1, 3))
          f_true_vec = np.reshape(f_true[index], (-1, 3))
          for i in range(num_entries // 3):
            px, py, pz = f_pred_vec[i]
            tx, ty, tz = f_true_vec[i]
            tf.logging.info(
              " * {:2d} Predicted: {: 9.6f} {: 9.6f} {: 9.6f}, True: {: 9.6f} "
              "{: 9.6f} {: 9.6f}".format(i, px, py, pz, tx, ty, tz)
            )

      # Save the y_true and y_pred to a npz file for plotting
      if FLAGS.run_once:
        np.savez("{}_at_{}.npz".format(FLAGS.dataset, global_step),
                 y_true=y_true, y_pred=y_pred, f_true=f_true, f_pred=f_pred)

      else:
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='MAE (eV) @ 1', simple_value=precision)
        summary.value.add(tag='R2 Score @ 1', simple_value=score)
        summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step


def evaluate(eval_dir):
  """Eval CIFAR-10 for a number of steps."""

  set_logging_configs(
    debug=False,
    logfile=join(eval_dir, FLAGS.logfile)
  )

  with tf.Graph().as_default() as graph:

    # Inference the KCNN model for evaluation
    y_calc, y_true, _, f_calc, f_true = kcnn_from_dataset(
      FLAGS.dataset,
      for_training=False
    )

    # Cast `y_true` to float32 and set the shape of the `y_nn` explicitly.
    y_true = tf.cast(y_true, tf.float32)
    y_calc.set_shape(y_true.get_shape().as_list())

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        VARIABLE_MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_dir = FLAGS.summary_dir or eval_dir
    if not tf.gfile.Exists(summary_dir):
      tf.gfile.MakeDirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir, graph)

    # Run the evalutions
    evaluated_steps = []

    while True:
      eval_at_step = eval_once(
        saver, summary_writer, y_true, y_calc, f_true, f_calc, summary_op
      )
      if len(evaluated_steps) > 0 and evaluated_steps[-1] != eval_at_step:
        evaluated_steps.clear()
      evaluated_steps.append(eval_at_step)
      if len(evaluated_steps) == FLAGS.stop_after_repeats:
        tf.logging.info("Automatically stop the evaluation after "
                        "{} repeats.".format(len(evaluated_steps)))
        break
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(_):
  """
  The main function.
  """
  eval_dir = get_eval_dir()
  if not tf.gfile.Exists(eval_dir):
    tf.gfile.MakeDirs(eval_dir)
  evaluate(eval_dir)


if __name__ == '__main__':
  tf.app.run()
