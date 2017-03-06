from __future__ import print_function

import numpy as np
import time
import h5py
import sys


def skewness(vector):
  """
  This function returns the cube root of the skewness of the given vector.

  Args:
    vector: a vector, [n, ]

  Returns:
    skewness: the skewness of the vector.

  References:
    http://en.wikipedia.org/wiki/Skewness
    http://en.wikipedia.org/wiki/Moment_%28mathematics%29

  """
  v = np.asarray(vector)
  sigma = np.std(v)
  s = np.mean((v - v.mean()) ** 3.0)
  eps = 1E-8
  if np.abs(sigma) < eps or np.abs(s) < eps:
    return 0.0
  else:
    return s / (sigma ** 3.0)


def get_usr_features(coords):
  """
  Return the USR feature vector of the given molecule.

  Args:
    coords: a flatten array of cartesian coordinates, [3N, ]

  Returns:
    usr_vector: the standard USR fingerprints.

  """

  def _compute_usr(v1, v2, v3, v4, c):
    vector = np.zeros(12)
    k = 0
    for v in [v1, v2, v3, v4]:
      di = np.linalg.norm(v - c, axis=1)
      vector[k: k + 3] = np.mean(di), np.std(di), skewness(di)
      k += 3
    return vector

  cart_coords = coords.reshape((-1, 3))
  x = cart_coords.mean(axis=0)
  d = np.linalg.norm(x - cart_coords, axis=1)
  y = cart_coords[np.argmin(d)]
  z = cart_coords[np.argmax(d)]
  d = np.linalg.norm(z - cart_coords, axis=1)
  w = cart_coords[np.argmax(d)]

  return _compute_usr(x, y, z, w, cart_coords)


def remove_duplicates(coords, energies, hdf5_file, threshold=0.995, verbose=True):
  """
  Remove duplicated structures. The similarity algorithm used here is USR.

  This implementation now takes about 15 minutes on my MacBook Pro using one
  core. The speed is acceptable.
  """
  if verbose:
    print("Remove duplicated data samples ...\n")

  group = "similarity"
  n = len(coords)

  hdb = h5py.File(hdf5_file)
  if group not in hdb:
    hdb.create_group(group)

  try:
    v = hdb[group]["usr"][:]
  except Exception:
    if verbose:
      print("Compute USR features ...")
    v = np.zeros((n, 12), dtype=np.float32)
    tic = time.time()
    for i in range(n):
      if verbose and i % 2000 == 0:
        sys.stdout.write("\rProgress: %7d  /  %7d" % (i, n))
      v[i] = get_usr_features(coords[i])
    if verbose:
      print("")
      print("Time for computing USR features: %.3f s\n" % (time.time() - tic))
    hdb[group].create_dataset("usr", data=v)

  try:
    indices = hdb[group]["indices"][:]
  except Exception:
    if verbose:
      print("Comparing similarities. Be patient ...\n")
    tic = time.time()
    keep = np.ones(n, dtype=bool)
    for i in range(n):
      if not keep[i]:
        continue
      sij = 1.0 / (1.0 + np.sum(np.abs(v[i] - v[i + 1:, ...]), axis=1) / 12.0)
      duplicates = np.where(sij > threshold)[0]
      if len(duplicates) > 0:
        keep[duplicates + i] = False
      if verbose and i % 1000 == 0:
        sys.stdout.write("\rProgress: %7d  /  %7d" % (i, n))
    indices = np.where(keep == False)[0]
    del keep
    if verbose:
      print("")
      print("Time for comparing similarities: %.3f s\n" % (time.time() - tic))
    hdb[group].create_dataset("indices", data=indices)
  finally:
    hdb.close()

  del v

  if verbose:
    print("Number of duplicated samples: %d\n" % len(indices))
  coords = np.delete(coords, indices, axis=0)
  energies = np.delete(energies, indices)
  return energies, coords


