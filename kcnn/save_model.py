#!coding=utf-8
"""
This script is used to convert raw trained models to simplified and flexible
versions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import json
import pipeline
from os.path import join
from kcnn import kcnn
from constants import GHOST, VARIABLE_MOVING_AVERAGE_DECAY
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("checkpoint_dir", "./events",
                           """The directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string("aux_nodes", None,
                           """Comma separated string as the names of the 
                           auxiliary nodes to expose.""")

# TODO: test the batch normalization


def _get_transformer_repr(configs):
  """
  Return the JSON string for creating a `MultiTransformer`.

  Args:
    configs: a `dict` of configurations read from a JSON file.

  Returns:
    js: a `str` as the JSON string representation of the parameters for creating
      a `MultiTransformer`.

  """
  atom_types = configs["species"]
  params = {"atom_types": [atom for atom in atom_types if atom != GHOST],
            "k_max": configs["k_max"],
            "norm_order": configs["norm_order"],
            "include_all_k": configs["include_all_k"],
            "periodic": configs["periodic"],
            "max_occurs": configs["max_occurs"],
            "atomic_forces": configs["atomic_forces_enabled"],
            "lj": configs.get("lj", False)}
  return json.dumps(params)


def _get_output_node_names(forces=False):
  """
  Return the names of the tensors that should be accessed.
  """
  tensors = ["kCON/Energy/Sum/1_and_k", "kCON/Energy/y_contribs",
             "kCON/one-body/weights", "kCON/Energy/one-body/Conv2D",
             "transformer/json", "placeholders/inputs", "placeholders/occurs",
             "placeholders/weights", "placeholders/split_dims"]

  if forces:
    tensors.extend(["placeholders/coefficients", "placeholders/indexing",
                    "kCON/Forces/forces"])

  return ",".join(tensors)


def get_tensors_to_restore():
  """
  Return a dict of (name, tensor_name) that should be restored from an exported
  model file.

  Returns:
    tensors: a `dict` as the tensors that should be restored.

  """
  return {"{}".format(name): "{}:0".format(name)
          for name in _get_output_node_names().split(",")}


def _inference(dataset_name, conv_sizes):
  """
  Inference a model of `KCNN` with inputs feeded from placeholders.

  Args:
    dataset_name: a `str` as the name of the dataset.
    conv_sizes: a `str` of comma-separated integers as the numbers of kernels.

  Returns:
    graph: a `tf.Graph` as the graph for inference.

  """
  graph = tf.Graph()

  with graph.as_default():

    configs = pipeline.get_configs(for_training=True, dataset_name=dataset_name)
    split_dims = configs["split_dims"]
    num_atom_types = configs["num_atom_types"]
    kbody_terms = [term.replace(",", "") for term in configs["kbody_terms"]]
    num_kernels = [int(units) for units in conv_sizes.split(",")]
    atomic_forces = configs["atomic_forces_enabled"]
    ck2 = configs["shape"][1]
    num_f, num_entries = configs["indexing_shape"]

    with tf.name_scope("transformer"):
      _ = tf.constant(_get_transformer_repr(configs), name="json")

    with tf.name_scope("placeholders"):
      inputs_ = tf.placeholder(
        tf.float32, shape=(None, 1, None, ck2), name="inputs")
      occurs_ = tf.placeholder(
        tf.float32, shape=(None, 1, 1, num_atom_types), name="occurs")
      weights_ = tf.placeholder(
        tf.float32, shape=(None, 1, None, 1), name="weights")
      split_dims_ = tf.placeholder(
        tf.int64, shape=(len(split_dims, )), name="split_dims")
      is_training_ = tf.placeholder(tf.bool, name="is_training")

      if atomic_forces:
        coef_ = tf.placeholder(
          tf.float32, shape=(None, None, ck2 * 6), name="coefficients")
        indexing_ = tf.placeholder(
          tf.int32, shape=(None, num_f, num_entries), name="indexing")
      else:
        coef_ = None
        indexing_ = None

    kcnn(inputs_, occurs_, weights_, is_training=is_training_,
         split_dims=split_dims_, num_atom_types=num_atom_types,
         kbody_terms=kbody_terms, num_kernels=num_kernels, verbose=False,
         atomic_forces=atomic_forces, coefficients=coef_, indexing=indexing_,
         add_summary=False)

  return graph


def save_model(checkpoint_dir, dataset, conv_sizes, verbose=True,
               auxiliary_outputs=None):
  """
  take a GraphDef proto, a SaverDef proto, and a set of variable values stored
  in a checkpoint file, and output a GraphDef with all of the variable ops
  converted into const ops containing the values of the variables.

  Args:
    checkpoint_dir: a `str` as the directory to look for trained metadata.
    dataset: a `str` as the name of dataset to use.
    conv_sizes: a `str` of comma-separated integers as the numbers of kernels.
    verbose: a `bool` indicating whether or not should log the progress.
    auxiliary_outputs: a `List[str]` as the of additional tensors to expose.

  See Also:
    https://www.tensorflow.org/extend/tool_developers

  """
  graph = _inference(dataset, conv_sizes)

  with tf.Session(graph=graph) as sess:

    # Restore the `moving averaged` model variables from the latest checkpoint.
    tf.global_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
      variable_averages = tf.train.ExponentialMovingAverage(
        VARIABLE_MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=1)
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = int(
        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      if verbose:
        print("Restore the latest checkpoint: {}".format(
          ckpt.model_checkpoint_path))

    else:
      raise IOError("Failed to restore the latest checkpoint!")

    # Make the directory for saving exported models
    freeze_dir = join(checkpoint_dir, "freeze")
    if not tf.gfile.Exists(freeze_dir):
      tf.gfile.MakeDirs(freeze_dir)

    # Save the current session to a checkpoint so that values of variables can
    # be restored later.
    # FIXME: find a universal solution for freezing graphs.
    if tf.__version__ >= "1.1":
      saver = tf.train.Saver(var_list=tf.trainable_variables())
    checkpoint_path = saver.save(
      sess,
      join(freeze_dir, "{}.ckpt".format(dataset)),
      global_step=global_step
    )

    # Write the graph to a GraphDef proto file
    graph_name = "{}-{}.pb".format(dataset, global_step)
    graph_path = join(freeze_dir, graph_name)
    graph_io.write_graph(graph, freeze_dir, name=graph_name)

    # Setup the configs and freeze the current graph
    output_node_names = _get_output_node_names()
    auxiliary_outputs = auxiliary_outputs or []
    if len(auxiliary_outputs) > 0:
      auxiliary_node_names = ",".join(auxiliary_outputs)
      output_node_names = ",".join([output_node_names, auxiliary_node_names])
    input_saver_def_path = ""
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = False
    input_binary = False
    freeze_graph.freeze_graph(graph_path, input_saver_def_path,
                              input_binary, checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, graph_path,
                              clear_devices, "")

    if verbose:
      print("Export the model to {}".format(graph_path))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def main(unused):
  if FLAGS.aux_nodes is not None:
    aux_nodes = [name.strip() for name in FLAGS.aux_nodes.split(",")]
  else:
    aux_nodes = None
  save_model(FLAGS.checkpoint_dir, FLAGS.dataset, FLAGS.conv_sizes,
             auxiliary_outputs=aux_nodes)


if __name__ == "__main__":
  tf.app.run(main=main)
