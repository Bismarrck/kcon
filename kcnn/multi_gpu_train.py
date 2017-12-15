# coding=utf-8
"""
This script is used to train the KCNN network using multiple GPUs.
"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import re
import time
import constants
import kcnn
import pipeline
from datetime import datetime
from os.path import join
from constants import LOSS_MOVING_AVERAGE_DECAY
from kcnn import extract_configs, BatchIndex
from kcnn import kcnn as inference
from save_model import save_model
from utils import set_logging_configs, save_training_flags
from summary_utils import add_total_norm_summaries

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

# Setup the I/O
tf.app.flags.DEFINE_string('train_dir', './events',
                           """The directory for storing training files.""")
tf.app.flags.DEFINE_string('logfile', "train.log",
                           """The training logfile.""")
tf.app.flags.DEFINE_integer('save_frequency', 200,
                            """The frequency, in number of global steps, that
                            the summaries are written to disk""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """The frequency, in number of global steps, that
                            the training progress wiil be logged.""")
tf.app.flags.DEFINE_integer('freeze_frequency', 0,
                            """The frequency, in number of global steps, that
                            the graph will be freezed and exported. Set this to
                            0 to disable freezing.""")
tf.app.flags.DEFINE_integer('max_to_keep', 100,
                            """The maximum number of checkpoints to keep.""")

# Setup the basic training parameters.
tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            """The maximum number of training epochs.""")
tf.app.flags.DEFINE_boolean('restore_training', True,
                            """Restore the previous training if possible.""")
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           """Start a new training using the variables restored
                           from the checkpoint.""")
tf.app.flags.DEFINE_boolean('forces_only', False,
                            """Only optimize force-related variables if this 
                            flag is set.""")

# Setup the devices.
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('first_gpu_id', 0,
                            """The id of the first gpu.""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Set the logging level to `logging.DEBUG`.""")


def tower_loss(batch, params, scope, reuse_variables=False):
  """Calculate the total loss on a single tower running the kCON model.

  Args:
    batch: a `tuple` of Tensors: 'inputs', 'y_true', 'occurs', 'weights',
      'y_weight', 'f_true', 'coefficients' and 'indexing'.
    params: a `dict` as the parameters for inference.
    scope: unique prefix string identifying the tower, e.g. 'tower0'
    reuse_variables: a `bool` indicating whether we should reuse the variables.

  Returns:
    Tensor of shape [] containing the total loss for a batch of data

  """

  # Inference the model of `kCON`
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    y_calc, f_calc, n_atom = inference(
      inputs=batch[BatchIndex.inputs],
      occurs=batch[BatchIndex.occurs],
      weights=batch[BatchIndex.weights],
      coefficients=batch[BatchIndex.coefficients],
      indexing=batch[BatchIndex.indexing],
      **params
    )

  # Cast the true values to float32 and set the shape of the `y_pred`
  # explicitly.
  y_true = tf.cast(batch[BatchIndex.y_true], tf.float32)
  y_calc.set_shape(y_true.get_shape().as_list())

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  if not FLAGS.forces:
    kcnn.get_y_loss(y_true, y_calc, weights=batch[BatchIndex.loss_weight])

  elif FLAGS.forces_only:
    f_true = batch[BatchIndex.f_true]
    kcnn.get_f_loss(f_true, f_calc)

  else:
    f_true = batch[BatchIndex.f_true]
    kcnn.get_yf_joint_loss(y_true, y_calc, f_true, f_calc)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(LOSS_MOVING_AVERAGE_DECAY,
                                                    name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s[0-9]*/' % constants.TOWER_NAME, '', l.op.name)
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
      if g is None:
        continue

      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    v = grad_and_vars[0][1]

    # If the grads are all None, we just return a None grad.
    if len(grads) == 0:
      grad_and_var = (None, v)

    else:
      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      grad_and_var = (grad, v)

    average_grads.append(grad_and_var)

  return average_grads


def get_splits(batch, num_splits):
  """
  Split the batch.

  Args:
    batch: a tuple of Tensors.
    num_splits: an `int` as the number of splits.

  Returns:
    tensors_splits: a list of tuple of input tensors for each tower.

  """
  inputs_splits = tf.split(
    num_or_size_splits=num_splits, value=batch[BatchIndex.inputs])
  occurs_splits = tf.split(
    num_or_size_splits=num_splits, value=batch[BatchIndex.occurs])
  y_true_splits = tf.split(
    num_or_size_splits=num_splits, value=batch[BatchIndex.y_true])
  weights_splits = tf.split(
    num_or_size_splits=num_splits, value=batch[BatchIndex.weights])
  y_weight_splits = tf.split(
    num_or_size_splits=num_splits, value=batch[BatchIndex.loss_weight])

  if FLAGS.forces:
    f_true_splits = tf.split(
      num_or_size_splits=num_splits, value=batch[BatchIndex.f_true])
    coef_splits = tf.split(
      num_or_size_splits=num_splits, value=batch[BatchIndex.coefficients])
    indexing_splits = tf.split(
      num_or_size_splits=num_splits, value=batch[BatchIndex.indexing])
  else:
    f_true_splits = [None] * num_splits
    coef_splits = [None] * num_splits
    indexing_splits = [None] * num_splits

  tensors_splits = []
  for i in range(num_splits):
    tensors_splits.append((
      inputs_splits[i], y_true_splits[i], occurs_splits[i], weights_splits[i],
      y_weight_splits[i], f_true_splits[i], coef_splits[i], indexing_splits[i]
    ))
  return tensors_splits


def restore_previous_checkpoint(sess, global_step):
  """
  Restore the moving averaged variables from a previous checkpoint.

  Args:
    sess: a `tf.Session`.
    global_step: the tensor of the `global_step`.

  """
  start_step = 0
  variable_averages = tf.train.ExponentialMovingAverage(
    constants.VARIABLE_MOVING_AVERAGE_DECAY)
  variables_to_restore = {}
  for var in tf.trainable_variables():
    variables_to_restore[variable_averages.average_name(var)] = var

  loader = tf.train.Saver(var_list=variables_to_restore)
  if FLAGS.restore_checkpoint:
    loader.restore(sess, FLAGS.restore_checkpoint)
  else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      loader.restore(sess, ckpt.model_checkpoint_path)
      start_step = sess.run(global_step)
  return start_step


def train_with_multiple_gpus():
  """
  Train the KCNN model with mutiple gpus.
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
    with tf.name_scope("Optimizer"):
      learning_rate = kcnn.get_learning_rate(global_step)
      opt = kcnn.get_optimizer(learning_rate)

    # Initialize the input pipeline.
    total_batch_size = FLAGS.batch_size * FLAGS.num_gpus
    num_examples = pipeline.get_dataset_size(FLAGS.dataset)
    batch = pipeline.next_batch(for_training=True,
                                shuffle=True,
                                dataset_name=FLAGS.dataset,
                                num_epochs=FLAGS.num_epochs,
                                batch_size=total_batch_size)
    configs = pipeline.get_configs(for_training=True)
    params = extract_configs(configs, for_training=True)

    # Split the batch for each tower
    tensors_splits = get_splits(batch, num_splits=FLAGS.num_gpus)

    # Retain all non-tower summaries
    non_tower_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

    # Calculate the gradients for each model tower.
    tower_grads = []
    summaries = []
    loss = None
    batchnorm_updates = []
    reuse_variables = False

    for i in range(FLAGS.first_gpu_id, FLAGS.first_gpu_id + FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s%d' % (constants.TOWER_NAME, i)) as scope:

          # Calculate the loss for one tower of the KCNN model.
          # This function constructs the entire model but shares the variables
          # across all towers.
          loss = tower_loss(tensors_splits[i], params, scope, reuse_variables)

          # Reuse variables for the next tower.
          reuse_variables = True

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Retain the Batch Normalization updates operations only from the
          # final tower. Ideally, we should grab the updates from all towers
          # but these stats accumulate extremely fast so we can ignore the
          # other stats from the other towers without significant detriment.
          if FLAGS.normalizer and FLAGS.normalizer == 'batch_norm':
            batchnorm_updates = tf.get_collection(
              tf.GraphKeys.UPDATE_OPS, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    summaries.extend(
      add_total_norm_summaries(grads, "yf", only_summary_total=False))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # Add histograms for gradients.
    with tf.name_scope("Summary"):
      for grad, var in grads:
        if grad is not None:
          summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))
      for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    with tf.name_scope("average"):
      variable_averages = tf.train.ExponentialMovingAverage(
        constants.VARIABLE_MOVING_AVERAGE_DECAY, global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    if FLAGS.normalizer and FLAGS.normalizer == 'batch_norm':
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      train_op = tf.group(
        batchnorm_updates_op, apply_gradient_op, variables_averages_op)
    else:
      train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Save the training flags
    save_training_flags(FLAGS.train_dir, dict(FLAGS.__dict__["__flags"]))

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries + non_tower_summaries)

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
    start_step = 0
    if FLAGS.restore_training or FLAGS.restore_checkpoint:
      start_step = restore_previous_checkpoint(sess, global_step)
    max_steps = int(FLAGS.num_epochs * num_examples / total_batch_size)

    # Create the summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(start_step, max_steps):
      start_time = time.time()

      try:
        _, loss_value = sess.run([train_op, loss])
      except tf.errors.OutOfRangeError:
        tf.logging.info(
          "Stop this training after {} epochs.".format(FLAGS.num_epochs))
        break

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % FLAGS.log_frequency == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus
        epoch = step * total_batch_size / num_examples
        format_str = "%s: step %6d, epoch=%7.2f, loss = %10.6f " \
                     "(%8.1f examples/sec; %8.3f sec/batch)"
        tf.logging.info(format_str % (datetime.now(), step, epoch, loss_value,
                                      examples_per_sec, sec_per_batch))

      if step % FLAGS.save_frequency == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % (20 // FLAGS.num_gpus * FLAGS.save_frequency) == 0 or \
              (step + 1) == max_steps:
        checkpoint_path = join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        tf.logging.info("{}-{} saved".format(checkpoint_path, step))

      if FLAGS.freeze_frequency > 0 and step > 0:
        if step % FLAGS.freeze_frequency == 0 or (step + 1) == max_steps:
          save_model(FLAGS.train_dir, FLAGS.dataset, FLAGS.conv_sizes)

    else:
      tf.logging.info('The maximum number of epochs already reached!')

    # Save the final model
    if FLAGS.freeze_frequency > 0:
      save_model(FLAGS.train_dir, FLAGS.dataset, FLAGS.conv_sizes)


def main(_):
  """
  The main function.
  """
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MkDir(FLAGS.train_dir)
  train_with_multiple_gpus()


if __name__ == '__main__':
  tf.app.run(main=main)
