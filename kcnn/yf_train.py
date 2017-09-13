# coding=utf-8
"""
This script is used to train kCON models on a single node with CPUs or a single
GPU for both the energy and the atomic forces.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import json
import time
from datetime import datetime
from constants import KcnnGraphKeys, LOSS_MOVING_AVERAGE_DECAY
from kcnn import kcnn_yf_from_dataset
from os.path import join
from utils import get_xargs, set_logging_configs


__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_float('beta1', 0.9,
                          """The beta1 of the AdamOptimizer.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """The maximum number of training steps.""")
tf.app.flags.DEFINE_integer('save_frequency', 25,
                            """The frequency, in number of global steps, that
                            the summaries are written to disk""")
tf.app.flags.DEFINE_integer('log_frequency', 5,
                            """The frequency, in number of global steps, that
                            the training progress wiil be logged.""")
tf.app.flags.DEFINE_integer('freeze_frequency', 0,
                            """The frequency, in number of global steps, that
                            the graph will be freezed and exported.""")
tf.app.flags.DEFINE_integer('max_to_keep', 50,
                            """The maximum number of checkpoints to keep.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('timeline', False,
                            """Enable timeline profiling if True.""")
tf.app.flags.DEFINE_string('logfile', "train.log",
                           """The training logfile.""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Set the logging level to `logging.DEBUG`.""")
tf.app.flags.DEFINE_boolean('restore', True,
                            """Restore the previous checkpoint if possible.""")


def save_training_flags():
  """
  Save the training flags to the train_dir.
  """
  args = dict(FLAGS.__dict__["__flags"])
  args["run_flags"] = " ".join(
    ["--{}={}".format(k, v) for k, v in args.items()]
  )
  cmdline = get_xargs()
  if cmdline:
    args["cmdline"] = cmdline
  with open(join(FLAGS.train_dir, "flags.json"), "w+") as f:
    json.dump(args, f, indent=2)


def _add_total_norm_summaries(grads_and_vars, collection,
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
      if not only_summary_total:
        with tf.name_scope("norms/{}/".format(collection)):
          tf.summary.scalar(var.op.name, norm)

  with tf.name_scope("total_norm/"):
    total_norm = tf.add_n(tf.get_collection(collection))
    tf.summary.scalar(collection, total_norm)

  return total_norm


def _add_loss_summaries(total_loss):
  """Add summaries for losses in KCNN model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().

  Returns:
    loss_averages_op: op for generating moving averages of losses.

  """

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(LOSS_MOVING_AVERAGE_DECAY,
                                                    name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + '_raw', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train_model():
  """
  Train the neural network model.
  """

  set_logging_configs(
    debug=FLAGS.debug,
    logfile=join(FLAGS.train_dir, FLAGS.logfile)
  )

  with tf.Graph().as_default():

    # Get the global step
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Inference the kCON model
    calc_tensors, true_tensors, auxiliary_tensors = kcnn_yf_from_dataset(
      dataset_name=FLAGS.dataset,
      for_training=True
    )

    # Summarize the histograms of the variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name + "/hist", var)

    # Get the tensors
    y_calc = calc_tensors["y"]
    f_calc = calc_tensors["f"]
    y_true = true_tensors["y"]
    f_true = true_tensors["f"]
    y_weight = auxiliary_tensors['y_weight']
    handles = auxiliary_tensors['handles']

    # Model loss
    with tf.name_scope("yfRMSE"):

      y_true = tf.cast(y_true, tf.float32)
      y_calc.set_shape(y_true.get_shape().as_list())

      with tf.name_scope("y_loss"):
        y_mse = tf.losses.mean_squared_error(
          y_true,
          y_calc,
          scope="MSE",
          loss_collection=None,
          weights=y_weight
        )
        rmse = tf.sqrt(y_mse, name="RMSE")
        tf.summary.scalar("yRMSE", rmse)
        tf.add_to_collection('y_losses', rmse)
        y_loss = tf.add_n(tf.get_collection('y_losses'), name='y_total_loss')

      with tf.name_scope("f_loss"):
        f_mse = tf.losses.mean_squared_error(
          f_true,
          f_calc,
          scope="fMSE",
          loss_collection=None,
        )
        f_rmse = tf.sqrt(f_mse, name="RMSE")
        tf.summary.scalar("fRMSE", f_rmse)
        f_loss = tf.multiply(f_rmse, FLAGS.floss_weight, name="sRMSE")
        tf.add_to_collection('f_losses', f_loss)
        f_loss = tf.add_n(tf.get_collection('f_losses'), name='f_total_loss')

    # Build the optimizers
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

      with tf.name_scope("Optimizers"):

        with tf.name_scope("y"):
          y_opt = tf.train.AdamOptimizer(
            FLAGS.learning_rate,
            name="AdamY",
            beta1=FLAGS.beta1
          )
          y_grads = y_opt.compute_gradients(
            y_loss,
            var_list=tf.get_collection(KcnnGraphKeys.ENERGY_VARIABLES)
          )
          _add_total_norm_summaries(
            y_grads,
            collection="y_norms",
            only_summary_total=False,
          )
          apply_y_grads_op = y_opt.apply_gradients(
            y_grads,
            global_step=global_step,
            name="apply_y_grads"
          )

        with tf.name_scope("f"):
          f_opt = tf.train.AdamOptimizer(
            FLAGS.f_learning_rate,
            name='AdamF',
            beta1=FLAGS.beta1
          )
          f_grads = f_opt.compute_gradients(
            f_loss,
            var_list=tf.get_collection(KcnnGraphKeys.FORCES_VARIABLES)
          )
          # _add_total_norm_summaries(
          #   f_grads,
          #   collection="f_norms",
          #   only_summary_total=False
          # )
          apply_f_grads_op = f_opt.apply_gradients(
            f_grads,
            global_step=global_step,
            name="apply_f_grads"
          )

    # Get the summary op
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

    # Save the training flags
    save_training_flags()

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=FLAGS.max_to_keep)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Initialize the dataset iterators
    dataset_handles = {}
    for key, dataset_iterator in auxiliary_tensors['dataset_iterators'].items():
      sess.run(dataset_iterator.initializer)
      dataset_handles[key] = sess.run(dataset_iterator.string_handle())

    # Restore the previous checkpoint
    start_step = 0
    if FLAGS.restore:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        start_step = sess.run(global_step)

    # Create the summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(start_step, FLAGS.max_steps):

      start_time = time.time()

      if step % FLAGS.save_frequency == 0:
        y_value, _, summary_str = sess.run(
          [y_loss, apply_y_grads_op, summary_op],
          feed_dict={handles['y']: dataset_handles['y']}
        )
        summary_writer.add_summary(summary_str, step)

      else:
        y_value, _ = sess.run(
          [y_loss, apply_y_grads_op],
          feed_dict={handles['y']: dataset_handles['y']}
        )

      f_value, _ = sess.run(
        [f_loss, apply_f_grads_op],
        feed_dict={handles['f']: dataset_handles['f']}
      )

      duration = time.time() - start_time

      if step % FLAGS.log_frequency == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        epoch = step * num_examples_per_step / (FLAGS.num_examples * 0.8)
        total_loss_val = y_value + f_value
        format_str = "%s: step %6d, epoch=%7.2f, loss=%10.6f, yloss=%10.6f, " \
                     "floss=%10.6f, (%8.1f examples/sec; %8.3f sec/batch)"
        tf.logging.info(
          format_str % (datetime.now(), step, epoch, total_loss_val, y_value,
                        f_value, examples_per_sec, duration))

      # Save the model checkpoint periodically.
      if step % (20 * FLAGS.save_frequency) == 0 or \
              (step + 1) == FLAGS.max_steps:
        checkpoint_path = join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(_):
  """
  The main function.
  """
  if not FLAGS.forces:
    print('This module only supports training energy and forces model.')

  else:
    if not tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.MkDir(FLAGS.train_dir)
    train_model()


if __name__ == "__main__":
  tf.app.run(main=main)
