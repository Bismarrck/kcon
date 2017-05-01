# coding=utf-8
"""
This script is used to train the sum-kbody-cnn network using multiple GPUs.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import kbody
import re
import time
import numpy as np
from datetime import datetime
from utils import get_xargs, set_logging_configs

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

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
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")


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


def tower_loss(scope):
  """Calculate the total loss on a single tower running the sum-kbody-cnn model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'

  Returns:
    Tensor of shape [] containing the total loss for a batch of data

  """
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

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = kbody.loss(y_true, y_pred)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % kbody.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  return average_grads


def train_with_multiple_gpus():
  """
  Train the sum-kbody-cnn model with mutiple gpus.
  """
  set_logging_configs(
    debug=FLAGS.debug,
    logfile=join(FLAGS.train_dir, FLAGS.logfile)
  )

  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
      'global_step',
      [],
      initializer=tf.constant_initializer(),
      trainable=False
    )

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)

    # Calculate the gradients for each model tower.
    tower_grads = []
    summaries = []
    loss = None
    assert FLAGS.num_gpus > 0

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (kbody.TOWER_NAME, i)) as scope:
            # Calculate the loss for one tower of the sum-kbody-cnn model.
            # This function constructs the entire model but shares the variables
            # across all towers.
            loss = tower_loss(scope)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

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
def main(argv=None):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MkDir(FLAGS.train_dir)
  train_with_multiple_gpus()


if __name__ == '__main__':
  tf.app.run()
