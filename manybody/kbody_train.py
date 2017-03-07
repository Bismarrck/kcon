# coding=utf-8
"""
This script is used to train the kbody network.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import time
import kbody
from datetime import datetime

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """The maximum number of training steps.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def train_model(*args):
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get features and energies.
    features, energies = kbody.inputs(train=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pred_energies, _ = kbody.inference(
      features,
      verbose=True
    )

    # Setup the loss function
    loss = kbody.get_total_loss(energies, pred_energies)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = kbody.get_train_op(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 100 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          fstr = ('%s: step %6d, loss = %10.6f (%6.1f examples/sec; %7.3f '
                  'sec/batch)')
          print(fstr % (datetime.now(), self._step, loss_value,
                        examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


if __name__ == "__main__":
  tf.app.run(main=train_model)
