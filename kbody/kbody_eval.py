# coding=-utf8
"""
Evaluation for k-body CNN.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from kbody import sum_kbody_cnn_from_dataset as inference
from utils import set_logging_configs
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from os.path import join

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './events/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './events',
                           """Directory where to read model checkpoints.""")
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


def eval_once(saver, summary_writer, y_true_op, y_pred_op, summary_op,
              feed_dict):
  """
  Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    y_true_op: The Tensor used for fetching true predictions.
    y_pred_op: The Tensor used for fetching neural network predictions.
    summary_op: Summary op.
    feed_dict: The dict used for `sess.run`.
  
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
      return

    # Wait until the parsed global step is not zero.
    if (not FLAGS.run_once) and global_step <= 1:
      tf.logging.info("The global step is <= 1. Wait ...")
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []

    try:
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_evals = FLAGS.num_evals
      num_iter = int(math.ceil(num_evals / FLAGS.batch_size))
      y_true = np.zeros((num_evals, ), dtype=np.float32)
      y_pred = np.zeros((num_evals, ), dtype=np.float32)
      step = 0
      while step < num_iter and not coord.should_stop():
        y_true_, y_pred_ = sess.run([y_true_op, y_pred_op], feed_dict=feed_dict)
        istart = step * FLAGS.batch_size
        istop = min(istart + FLAGS.batch_size, num_evals)
        y_true[istart: istop] = -y_true_
        y_pred[istart: istop] = -y_pred_
        step += 1

      # Compute the common evaluation metrics.
      precision = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      y_diff = np.abs(y_true - y_pred)
      emax = y_diff.max()
      emin = y_diff.min()
      score = r2_score(y_true, y_pred)

      dtime = datetime.now()
      tf.logging.info("%s: step      = %d" % (dtime, global_step))
      tf.logging.info("%s: precision = %10.6f" % (dtime, precision))
      tf.logging.info("%s: RMSE      = %10.6f" % (dtime, rmse))
      tf.logging.info("%s: minimum   = %10.6f" % (dtime, emin))
      tf.logging.info("%s: maximum   = %10.6f" % (dtime, emax))
      tf.logging.info("%s: score     = %.6f" % (dtime, score))

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

      # Save the y_true and y_pred to a npz file for plotting
      if FLAGS.run_once:
        np.savez("{}_at_{}.npz".format(FLAGS.dataset, global_step),
                 y_true=y_true, y_pred=y_pred)

      else:
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op, feed_dict=feed_dict))
        summary.value.add(tag='MAE (eV) @ 1', simple_value=precision)
        summary.value.add(tag='R2 Score @ 1', simple_value=score)
        summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""

  set_logging_configs(
    debug=False,
    logfile=join(FLAGS.eval_dir, FLAGS.logfile)
  )

  with tf.Graph().as_default() as graph:

    # Inference the model of `sum-kbody-cnn` for evaluation
    y_nn, y_true, _, feed_dict = inference(FLAGS.dataset, for_training=False)
    y_true = tf.cast(y_true, tf.float32)
    y_nn.set_shape(y_true.get_shape().as_list())

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        kbody.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph)

    while True:
      eval_once(saver, summary_writer, y_true, y_nn, summary_op, feed_dict)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


# pylint: disable=unused-argument
# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(argv=None):
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
