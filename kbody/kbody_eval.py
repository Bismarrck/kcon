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
import kbody
from sklearn.metrics import r2_score

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


def eval_once(saver, summary_writer, y_true_op, y_pred_op, mae_op, summary_op,
              feed_dict):
  """
  Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    y_true_op: The Tensor used for fetching true predictions.
    y_pred_op: The Tensor used for fetching neural network predictions.
    mae_op: The `mean-absolute-error` op.
    summary_op: Summary op.
    feed_dict: The dict used for `sess.run`.
  
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []

    try:
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_evals / FLAGS.batch_size))
      maes = np.zeros((num_iter, ), dtype=np.float32)
      y_true = np.zeros((FLAGS.num_evals, ), dtype=np.float32)
      y_pred = np.zeros((FLAGS.num_evals, ), dtype=np.float32)
      step = 0
      while step < num_iter and not coord.should_stop():
        mae_val, y_true_, y_pred_ = sess.run(
          [mae_op, y_true_op, y_pred_op],
          feed_dict=feed_dict
        )
        maes[step] = mae_val
        istart = step * FLAGS.batch_size
        istop = min(istart + FLAGS.batch_size, FLAGS.num_evals)
        y_true[istart: istop] = y_true_
        y_pred[istart: istop] = y_pred_
        step += 1

      # Compute the Mean-Absolute-Error @ 1.
      precision = maes.mean()
      print('%s: precision = %10.6f' % (datetime.now(), precision))

      # Compute the linear coefficient
      score = r2_score(y_true, y_pred)
      print(" * R2 score: ", score)

      # Randomly output 10 predictions and true values
      indices = np.random.choice(range(FLAGS.num_evals), size=10)
      for i in indices:
        print(" * Predicted: %10.6f,  Real: %10.6f" % (y_pred[i], y_true[i]))

      # Save the y_true and y_pred to a npz file for plotting
      if FLAGS.run_once:
        np.savez("eval.npz", y_true=y_true, y_pred=y_pred)

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
  with tf.Graph().as_default() as g:

    # Read dataset configurations
    settings = kbody.inputs_settings(train=False)
    split_dims = settings["split_dims"]
    nat = settings["nat"]
    kbody_terms = [x.replace(",", "") for x in settings["kbody_terms"]]

    # Get features and energies for evaluation.
    batch_inputs, batch_true, batch_occurs, batch_weights = kbody.inputs(
      train=False, shuffle=True,
    )

    # Build a Graph that computes the logits predictions from the
    # inference model.
    batch_split_dims = tf.placeholder(
      tf.int64, [len(split_dims), ], name="split_dims"
    )

    # Parse the convolution layer sizes
    conv_sizes = [int(x) for x in FLAGS.conv_sizes.split(",")]
    if len(conv_sizes) < 2:
      raise ValueError("At least three convolution layers are required!")

    y_pred, _ = kbody.inference(
      batch_inputs,
      batch_occurs,
      batch_weights,
      nat=nat,
      split_dims=batch_split_dims,
      kbody_terms=kbody_terms,
      verbose=True,
      conv_sizes=conv_sizes
    )
    y_true = tf.cast(batch_true, tf.float32)

    # Calculate predictions.
    mae_op = tf.losses.absolute_difference(y_true, y_pred)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        kbody.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    # Build the feed dict
    feed_dict = {batch_split_dims: split_dims}

    while True:
      eval_once(saver, summary_writer, y_true, y_pred, mae_op,
                summary_op, feed_dict)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


# pylint: disable=unused-argument
# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(argv=None):
  evaluate()


if __name__ == '__main__':
  tf.app.run()
