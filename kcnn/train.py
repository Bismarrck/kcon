# coding=utf-8
"""
This script is used to train the kCON model using a single node with CPUs or
a single GPU.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import time
import kcnn
import pipeline
from kcnn import kcnn_from_dataset
from save_model import save_model
from os.path import join
from tensorflow.python.client.timeline import Timeline
from utils import set_logging_configs, save_training_flags

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            """The maximum number of training epochs.""")
tf.app.flags.DEFINE_integer('save_frequency', 200,
                            """The frequency, in number of global steps, that
                            the summaries are written to disk""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
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
tf.app.flags.DEFINE_boolean('forces_only', False,
                            """Only minimize the forces if this flag is set.""")
tf.app.flags.DEFINE_boolean('alter_train_op', False,
                            """Use the alternative training op if this flag is 
                            set.""")


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

    # Inference the kCON energy model
    y_calc, y_true, y_weights, f_calc, f_true = kcnn_from_dataset(
      FLAGS.dataset,
      for_training=True,
      num_epochs=FLAGS.num_epochs
    )

    # Cast `y_true` to float32 and set the shape of the `y_nn` explicitly.
    y_true = tf.cast(y_true, tf.float32)
    y_calc.set_shape(y_true.get_shape().as_list())

    # Setup the loss function
    y_loss = None
    f_loss = None

    if not FLAGS.forces:
      total_loss = kcnn.get_y_loss(y_true, y_calc, y_weights)

    elif FLAGS.forces_only:
      total_loss = kcnn.get_f_loss(f_true, f_calc)

    else:
      total_loss, y_loss, f_loss = kcnn.get_yf_joint_loss(
        y_true, y_calc, f_true, f_calc
      )

    # Build a Graph that trains the model.
    if FLAGS.forces and FLAGS.alter_train_op:
      train_op = kcnn.get_yf_train_op(y_loss, f_loss, global_step)
    else:
      train_op = kcnn.get_joint_loss_train_op(total_loss, global_step)

    # Save the training flags
    save_training_flags(FLAGS.train_dir, dict(FLAGS.__dict__["__flags"]))

    # Get the total number of training examples
    num_examples = pipeline.get_dataset_size(FLAGS.dataset)
    max_steps = int(num_examples * FLAGS.num_epochs / FLAGS.batch_size)

    class RunHook(tf.train.SessionRunHook):
      """ Log loss and runtime and regularly freeze the model. """

      def __init__(self, atomic_forces=False, should_freeze=True):
        """
        Initialization method.
        """
        super(RunHook, self).__init__()
        self._step = -1
        self._start_time = 0
        self._epoch = 0.0
        self._epoch_per_step = FLAGS.batch_size / num_examples
        self._log_frequency = FLAGS.log_frequency
        self._should_freeze = should_freeze
        self._freeze_frequency = FLAGS.freeze_frequency
        self._atomic_forces = atomic_forces

      def begin(self):
        """
        Called once before using the session.
        """
        self._step = -2

      def before_run(self, run_context):
        """
        Called before each call to run().

        Args:
          run_context: a `tf.train.SessionRunContext` as the context to execute
            ops and tensors.

        Returns:
          args: a `tf.train.SessionRunArgs` as the ops and tensors to execute
            under `run_context`.

        """
        self._step += 1
        self._epoch = self._step * self._epoch_per_step
        self._start_time = time.time()

        if not self._atomic_forces or FLAGS.forces_only:
          return tf.train.SessionRunArgs({"loss": total_loss,
                                          "global_step": global_step})
        else:
          return tf.train.SessionRunArgs({"loss": total_loss,
                                          "y_loss": y_loss,
                                          "f_loss": f_loss,
                                          "global_step": global_step})

      def should_log(self):
        """
        Return True if we should log the stats of current step.
        """
        return self._step % self._log_frequency == 0

      def should_freeze(self):
        """
        Return True if we should freeze the current graph and values.
        """
        return self._should_freeze and self._step % self._freeze_frequency == 0

      def after_run(self, run_context, run_values):
        """
        Called after each call to run().

        Args:
          run_context: a `tf.train.SessionRunContext` as the context to execute
           ops and tensors.
          run_values: results of requested ops/tensors by `before_run()`.

        """
        if self._step < 0:
          self._step = run_values.results["global_step"]

        duration = time.time() - self._start_time
        loss_value = run_values.results["loss"]
        num_examples_per_step = FLAGS.batch_size

        if self.should_log():
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          if not self._atomic_forces or FLAGS.forces_only:
            format_str = "step %6d, epoch=%7.2f, loss=%10.6f " \
                       "(%6.1f examples/sec; %7.3f sec/batch)"
            tf.logging.info(
              format_str % (self._step, self._epoch, loss_value,
                            examples_per_sec, sec_per_batch)
            )
          else:
            y_val = run_values.results['y_loss']
            f_val = run_values.results['f_loss']

            format_str = "step %6d, epoch=%7.2f, loss=%10.6f, y_loss=%10.6f, " \
                         "f_loss = %10.6f (%6.1f examples/sec; %7.3f sec/batch)"
            tf.logging.info(
              format_str % (self._step, self._epoch, loss_value, y_val, f_val,
                            examples_per_sec, sec_per_batch)
            )

        if self.should_freeze():
          save_model(FLAGS.train_dir, FLAGS.dataset, FLAGS.conv_sizes)

    run_meta = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    scaffold = tf.train.Scaffold(
      saver=tf.train.Saver(max_to_keep=FLAGS.max_to_keep))

    # noinspection PyMissingOrEmptyDocstring
    class TimelineHook(tf.train.SessionRunHook):
      """ A hook to output tracing results for further performance analysis. """

      def __init__(self):
        super(TimelineHook, self).__init__()
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

    export_graph = True if FLAGS.freeze_frequency else False

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        save_summaries_steps=FLAGS.save_frequency,
        hooks=[RunHook(should_freeze=export_graph,
                       atomic_forces=FLAGS.forces),
               TimelineHook(),
               tf.train.StopAtStepHook(last_step=max_steps)],
        scaffold=scaffold,
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      while not mon_sess.should_stop():
        try:
          if FLAGS.timeline:
            mon_sess.run(
              train_op, options=run_options, run_metadata=run_meta
            )
          else:
            mon_sess.run(train_op)
        except tf.errors.OutOfRangeError:
          tf.logging.info(
            "Stop this training after {} epochs.".format(FLAGS.num_epochs))
          break

  # Do not forget to export the final model
  if export_graph:
    save_model(FLAGS.train_dir, FLAGS.dataset, FLAGS.conv_sizes)


def main(_):
  """
  The main function.
  """
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MkDir(FLAGS.train_dir)
  train_model()


if __name__ == "__main__":
  tf.app.run(main=main)
