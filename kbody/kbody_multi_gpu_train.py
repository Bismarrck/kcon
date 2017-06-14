# coding=utf-8
"""
This script is used to train the sum-kbody-cnn network using multiple GPUs.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import re
import time
from kbody import BatchIndex
from kbody import extract_configs, get_batch, get_batch_configs
from kbody import sum_kbody_cnn as inference
from datetime import datetime
from os.path import join
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
tf.app.flags.DEFINE_string('logfile', "train.log",
                           """The training logfile.""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Set the logging level to `logging.DEBUG`.""")
tf.app.flags.DEFINE_integer('max_to_keep', None,
                            """The maximum number of checkpoints to keep.""")
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


def tower_loss(batch, params, scope, reuse_variables=False):
  """Calculate the total loss on a single tower running the sum-kbody-cnn model.

  Args:
    batch: a `tuple` of Tensors as the `inputs`, `y_true`, `occurs`, `weights`
      and `y_weight`.
    params: a `dict` as the parameters for inference.
    scope: unique prefix string identifying the tower, e.g. 'tower0'
    reuse_variables: a `bool` indicating whether we should reuse the variables.

  Returns:
    Tensor of shape [] containing the total loss for a batch of data

  """

  # Inference the model of `sum-kbody-cnn`
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    y_nn = inference(batch[BatchIndex.inputs], batch[BatchIndex.occurs],
                     batch[BatchIndex.weights], **params)

  # Cast the true values to float32 and set the shape of the `y_pred`
  # explicitly.
  y_true = tf.cast(batch[BatchIndex.y_true], tf.float32)
  y_nn.set_shape(y_true.get_shape().as_list())

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  kbody.loss(y_true, y_nn, weights=batch[BatchIndex.loss_weight])

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub(r'%s[0-9]*/' % kbody.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
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

    # Get or create the global step variable to count the number of train()
    # calls. This equals the number of batches processed * FLAGS.num_gpus.
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # Initialize the input pipeline
    batch = get_batch(train=True, shuffle=True)
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
      batch, capacity=2 * FLAGS.num_gpus)
    configs = get_batch_configs(train=True)
    params, feed_dict = extract_configs(configs)

    # Calculate the gradients for each model tower.
    tower_grads = []
    summaries = []
    loss = None
    reuse_variables = False
    assert FLAGS.num_gpus > 0

    for i in range(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s%d' % (kbody.TOWER_NAME, i)) as scope:
          # Dequeues one batch for the GPU
          gpu_batch = batch_queue.dequeue()

          # Calculate the loss for one tower of the sum-kbody-cnn model.
          # This function constructs the entire model but shares the variables
          # across all towers.
          loss = tower_loss(gpu_batch, params, scope, reuse_variables)

          # Reuse variables for the next tower.
          reuse_variables = True

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
      kbody.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Save the training flags
    save_training_flags()

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

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

    # Restore the previous checkpoint
    init_step = 0
    if FLAGS.restore:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = sess.run(global_step)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # Create the summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(init_step, FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % FLAGS.log_frequency == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus
        epoch = step * num_examples_per_step / (FLAGS.num_examples * 0.8)
        format_str = "%s: step %6d, epoch=%7.2f, loss = %10.6f " \
                     "(%8.1f examples/sec; %8.3f sec/batch)"
        tf.logging.info(format_str % (datetime.now(), step, epoch, loss_value,
                                      examples_per_sec, sec_per_batch))

      if step % FLAGS.save_frequency == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % (20 // FLAGS.num_gpus * FLAGS.save_frequency) == 0 or \
              (step + 1) == FLAGS.max_steps:
        checkpoint_path = join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(argv=None):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MkDir(FLAGS.train_dir)
  train_with_multiple_gpus()


if __name__ == '__main__':
  tf.app.run()
