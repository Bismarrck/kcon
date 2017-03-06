from __future__ import print_function, absolute_import

import numpy as np
import re
import sys
import time
import symmetry
import tensorflow as tf
from os.path import join, isfile
from sklearn.model_selection import train_test_split

__author__ = 'Xin Chen'
__email__ = "chenxin13@mails.tsinghua.edu.cn"


# The number of atoms is 20
NATOMS = 20

# Use the default Behler settings, so the number of features for each atom is 8,
# including 4 radial features and 4 angular features.
NDIMS = 8

# The cutoff radius is set to 6.5 angstroms.
Rc = 6.5

# Use the optimized B20 structures to train this demo
XYZ_FILE = join("..", "datasets", "B20pbe_opted.xyz")

# The total number of training samples.
TOTAL_SIZE = 2400

# Use 20% of the samples as the testing samples.
TEST_SIZE = 0.2

# Save the npz as the intermediate file.
NPZ_FILE = "B20.npz"

# Set the TensorFlowRecords file to save fingerprints
TRAIN_FILE = "B20train.tfrecords"
TEST_FILE = "B20test.tfrecords"

# Set the seed
SEED = 218


def extract_xyz(filename, verbose=True):
  """
  Extract symbols, coordiantes and forces (for later usage) from the file.

  Args:
    filename: a str, the file to parse.
    verbose: a bool.

  Returns
    species: Array[NUM_SITES], an array of the atomic symbols.
    energies: Array[N,], a 1D array of the atomic energies.
    coordinates: Array[N, 17, 3], a 3D array of the atomic coordinates.

  """
  energies = np.zeros((TOTAL_SIZE,), dtype=np.float32)
  coordinates = np.zeros((TOTAL_SIZE, NATOMS, 3), dtype=np.float32)
  species = []
  parse_species = True
  parse_forces = False
  stage = 0
  i = 0
  j = 0

  energy_patt = re.compile(r"([\w.-]+)")
  string_patt = re.compile(
    r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")

  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")

  with open(filename) as f:
    for line in f:
      if i == TOTAL_SIZE:
        break
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          n = int(l)
          if n != NATOMS:
            raise ValueError("The parsed size %d != NUM_SITES" % n)
          stage += 1
      elif stage == 1:
        m = energy_patt.search(l)
        if m:
          energies[i] = float(m.group(1))
          stage += 1
      elif stage == 2:
        m = string_patt.search(l)
        if m:
          coordinates[i, j, :] = float(m.group(2)), float(m.group(3)), float(
            m.group(4))
          if parse_species:
            species.append(m.group(1))
            if len(species) == NATOMS:
              species = np.asarray(species, dtype=object)
              parse_species = False
          j += 1
          if j == NATOMS:
            j = 0
            stage = 0
            i += 1
            if verbose and i % 1000 == 0:
              sys.stdout.write("\rProgress: %7d  /  %7d" % (i, TOTAL_SIZE))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  return species, energies, coordinates


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_to_records(features, energies, filename):
  """
  Write the Numpy arrays of `features` and `energies` to tfrecords.

  Args:
    features: a `[-1, N, M]` array as the Behler features.
    energies: a `[-1, 1]` array as the desired energies.
    filename: a stirng, the file to write.

  """
  ntotal, height, depth = features.shape[:]
  width = 1
  features = np.reshape(features, (-1, width, height, depth))
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(ntotal):
    x = features[index].tostring()
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
          'energy': _float_feature(energies[index]),
          'behlers': _bytes_feature(x)
      }))
    writer.write(example.SerializeToString())
  writer.close()


def inverse_transform(x, energy=True):
  """
  Inverse transform the scaled features or energies.

  Args:
    x: an `np.ndarray`.
    energy: boolean indicating the given `x` represents features or energies.

  Returns:
    y: an inversed array.

  """
  if not isfile(NPZ_FILE):
    raise ValueError("The dataset may not be built or accessed!")
  ar = np.load(NPZ_FILE)
  scales = ar["scales"]
  if energy:
    lmax, lmin = scales[-1]
    return lmin + x * (lmax - lmin)
  else:
    # Shall be fixed!
    return x


def may_build_dataset(verbose=True):
  """
  Build the dataset if the npz file cannot be accessed. Then return the energies
  and Behler features.

  Args:
    verbose: a bool, If True the running progress will be printed.

  Returns:
    features: a 3D array as the Behler's training features.
    energies: a 2D array as the desired energies.

  """
  if not isfile(NPZ_FILE):

    _, energies, coordinates = extract_xyz(XYZ_FILE, verbose=verbose)
    ntotal = len(energies)
    features = np.zeros((ntotal, NATOMS, NDIMS), dtype=np.float32)
    energies = energies.astype(np.float32)

    print("Building Behler's fingerprints ...")
    tic = time.time()

    for i in range(ntotal):
      features[i] = symmetry.get_behler_fingerprints(coordinates[i], Rc)
      if verbose and i % 100 == 0:
        sys.stdout.write("\rProgress: %7d / %7d" % (i, ntotal))
    print("")
    print("Total time: %.3f s" % (time.time() - tic))
    print("")

    # Scale the energies and features to [0, 1].
    scales = np.zeros((NDIMS + 1, 2))
    for l in range(NDIMS):
      lmax, lmin = features[:, :, l].max(), features[:, :, l].min()
      features[:, :, l] = (features[:, :, l] - lmin) / (lmax - lmin)
      scales[l] = lmin, lmax

    lmax, lmin = energies.max(), energies.min()
    energies = (energies - lmin) / (lmax - lmin)
    scales[-1] = lmax, lmin

    np.savez(NPZ_FILE, features=features, energies=energies, scales=scales)

  else:
    ar = np.load(NPZ_FILE)
    features = ar["features"]
    energies = ar["energies"]

  X_train, X_test, y_train, y_test = train_test_split(
    features,
    energies,
    random_state=SEED,
    test_size=TEST_SIZE
  )

  write_to_records(X_train, y_train, TRAIN_FILE)
  write_to_records(X_test, y_test, TEST_FILE)


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'behlers': tf.FixedLenFeature([], tf.string),
          'energy': tf.FixedLenFeature([], tf.float32),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  behlers = tf.decode_raw(features['behlers'], tf.float32)
  behlers.set_shape([NATOMS * NDIMS])
  behlers = tf.reshape(behlers, [1, NATOMS, NDIMS])
  energy = features['energy']

  return behlers, energy


def inputs(train, batch_size, num_epochs, shuffle=True):
  """
  Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    shuffle:

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

  """
  if not num_epochs:
    num_epochs = None

  filename = TRAIN_FILE if train else TEST_FILE

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename],
      num_epochs=num_epochs
    )

    # Even when reading in multiple threads, share the filename
    # queue.
    behlers, energy = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if shuffle:
      features, energies = tf.train.shuffle_batch(
        [behlers, energy], batch_size=batch_size, num_threads=8,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    else:
      features, energies = tf.train.batch(
        [behlers, energy], batch_size=batch_size, num_threads=8,
        capacity=1000 + 3 * batch_size)
    return features, energies


def test_read_and_decode():

  with tf.Session() as sess:

    features, energies = inputs(True, 1, 1, shuffle=False)
    f = sess.run([features])
    print(f)


if __name__ == "__main__":
  # may_build_dataset(verbose=True)
  test_read_and_decode()
