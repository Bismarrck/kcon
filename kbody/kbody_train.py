# coding=utf-8
"""
This script is used to train the kbody network.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import json
import time
import kbody
from datetime import datetime
from os.path import join
from tensorflow.python.client.timeline import Timeline
from utils import get_xargs, set_logging_configs

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """The maximum number of training steps.""")
tf.app.flags.DEFINE_integer('save_frequency', 200,
                            """The frequency, in number of global steps, that
                            the summaries are written to disk""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """The frequency, in number of global steps, that
                            the training progress wiil be logged.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('timeline', False,
                            """Enable timeline profiling if True.""")
tf.app.flags.DEFINE_string('logfile', "train.log",
                           """The training logfile.""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Set the logging level to `logging.DEBUG`.""")
tf.app.flags.DEFINE_integer('max_to_keep', None,
                            """The maximum number of checkpoints to keep.""")


def _save_training_flags():
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

    # Read dataset configurations
    settings = kbody.inputs_settings(train=True)
    split_dims = settings["split_dims"]
    nat = settings["nat"]
    kbody_terms = [x.replace(",", "") for x in settings["kbody_terms"]]
    initial_one_body_weights = settings["initial_one_body_weights"]

    # Get features and energies.
    batch_inputs, batch_true, batch_occurs, batch_weights = kbody.inputs(
      train=True
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

    # Inference
    y_pred, _ = kbody.inference(
      batch_inputs,
      batch_occurs,
      batch_weights,
      nat=nat,
      split_dims=batch_split_dims,
      kbody_terms=kbody_terms,
      conv_sizes=conv_sizes,
      initial_one_body_weights=np.asarray(initial_one_body_weights[:-1]),
      verbose=True,
    )

    # Cast the true values to float32 and set the shape of the `y_pred`
    # explicitly.
    y_true = tf.cast(batch_true, tf.float32)
    y_pred.set_shape(y_true.get_shape().as_list())

    # Setup the loss function
    loss = kbody.loss(y_true, y_pred)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = kbody.get_train_op(loss, global_step)

    # Save the training flags
    _save_training_flags()

    # noinspection PyMissingOrEmptyDocstring
    class _LoggerHook(tf.train.SessionRunHook):
      """ Logs loss and runtime."""

      def __init__(self):
        super(_LoggerHook, self).__init__()
        self._step = -1
        self._start_time = 0
        self._epoch = 0.0
        self._log_frequency = FLAGS.log_frequency

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._epoch = self._step / (FLAGS.num_examples * 0.8 / FLAGS.batch_size)
        self._start_time = time.time()
        return tf.train.SessionRunArgs({"loss": loss})

      def should_log(self):
        return self._step % self._log_frequency == 0

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results["loss"]
        num_examples_per_step = FLAGS.batch_size
        if self.should_log():
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          format_str = "%s: step %6d, epoch=%7.2f, loss = %10.6f " \
                       "(%6.1f examples/sec; %7.3f sec/batch)"
          tf.logging.info(
            format_str % (datetime.now(), self._step, self._epoch, loss_value,
                          examples_per_sec, sec_per_batch)
          )

    run_meta = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    scaffold = tf.train.Scaffold(
      saver=tf.train.Saver(max_to_keep=FLAGS.max_to_keep))

    # noinspection PyMissingOrEmptyDocstring
    class _TimelineHook(tf.train.SessionRunHook):
      """ A hook to output tracing results for further performance analysis. """

      def __init__(self):
        super(_TimelineHook, self).__init__()
        self._counter = -1

      def begin(self):
        self._counter = -1

      def get_ctf(self):
        return join(FLAGS.train_dir, "prof_%d.json" % self._counter)

      def should_save(self):
        return FLAGS.timeline and self._counter % FLAGS.save_frequency == 0

      def after_run(self, run_context, run_values):
        self._counter += 1
        if self.should_save():
          timeline = Timeline(step_stats=run_meta.step_stats)
          ctf = timeline.generate_chrome_trace_format(show_memory=True)
          with open(self.get_ctf(), "w+") as f:
            f.write(ctf)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_summaries_steps=FLAGS.save_frequency,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook(),
               _TimelineHook()],
        scaffold=scaffold,
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      feed_dict = {batch_split_dims: split_dims}

      while not mon_sess.should_stop():
        if FLAGS.timeline:
          mon_sess.run(
            train_op,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_meta
          )
        else:
          mon_sess.run(train_op, feed_dict=feed_dict)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MkDir(FLAGS.train_dir)
  train_model()


if __name__ == "__main__":
  tf.app.run(main=main)
