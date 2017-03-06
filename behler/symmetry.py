from __future__ import print_function, absolute_import

import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import product

__author__ = "Xin Chen"
__email__ = "Bismarrck@me.com"


def get_neighbour_list(r, cutoff):
  """
  A naive implementation of calculating neighbour list for non-periodic
  molecules.

  Args:
    r: a `[N, N]` array as the pairwise interatomic distances.
    cutoff: a float as the cutoff radius.

  Returns:
    neighbours: a `[N, N]` boolean array as the neighbour results.

  """
  neighbours = r < cutoff
  np.fill_diagonal(neighbours, False)
  return neighbours


def cutoff_fxn(r, rc):
  """
  The cutoff function.

  Args:
    r: a `float` or an array as the interatomic distances.
    rc: a `float` as the cutoff radius.

  Returns:
    fr: the damped `r`.

  """
  return (np.cos(np.minimum(r / rc, 1.0) * np.pi) + 1.0) * 0.5


def get_behler_fingerprints(coords, rc, radial_etas=None, angular_etas=None,
                            gammas=None, zetas=None):
  """
  Return the Behler's fingerprints.

  Args:
    coords: a `[N, 3]` array as the atomic coordinates.
    rc: a float as the cutoff radius.
    radial_etas: a `List[float]` as the `eta` in the radial functions.
    angular_etas: a `List[float]` as the `eta` in the angular functions.
    gammas: a `List[float]` as the `lambda` in the angular functions.
    zetas: a `List[float]` as the `zeta` in the angular functions.

  Returns:
    fingerprints: a `[N, M]` array as the fingerprints of the given molecule.

  """

  # These are default parameters defined by Behler. I just copied these from the
  # Amp source file `gaussian.py`.
  if radial_etas is None:
    radial_etas = [0.05, 4., 20., 80.]
  if angular_etas is None:
    angular_etas = [0.005]
  if gammas is None:
    gammas = [+1., -1.]
  if zetas is None:
    zetas = [1., 4.]

  r = pairwise_distances(coords)
  radials = _get_radial_fingerprints(
    coords,
    r,
    rc,
    radial_etas
  )
  augulars = _get_augular_fingerprints_naive(
    coords,
    r,
    rc,
    angular_etas,
    gammas, zetas
  )
  return np.hstack((radials, augulars))


def _get_radial_fingerprints(coords, r, rc, etas):
  """
  Return the fingerprints from the radial gaussian functions.

  Args:
    coords: a `[N, 3]` array as the cartesian coordinates.
    r: a `[N, N]` array as the pairwise distance matrix.
    rc: a float as the cutoff radius.
    etas: a `List[float]` as the `eta` in the radial functions.

  Returns:
    x: a `[N, M]` array as the radial fingerprints.

  """

  params = np.array(etas)
  ndim = len(params)
  natoms = len(coords)
  x = np.zeros((natoms, ndim))
  nl = get_neighbour_list(r, rc)
  rc2 = rc ** 2
  fr = cutoff_fxn(r, rc)

  for l, eta in enumerate(etas):
    for i in range(natoms):
      v = 0.0
      ri = coords[i]
      for j in range(natoms):
        if not nl[i, j]:
          continue
        rs = coords[j]
        ris = np.sum(np.square(ri - rs))
        v += np.exp(-eta * ris / rc2) * fr[i, j]
      x[i, l] = v
  return x


def _get_augular_fingerprints_naive(coords, r, rc, etas, gammas, zetas):
  """
  Return the fingerprints from the augular functions.

  Args:
    coords: a `[N, 3]` array as the cartesian coordinates.
    r: a `[N, N]` matrix as the pairwise distances.
    rc: a float as the cutoff radius.
    etas: a `List[float]` as the `eta` in the radial functions.
    gammas: a `List[float]` as the `lambda` in the angular functions.
    zetas: a `List[float]` as the `zeta` in the angular functions.

  Returns:
    x: a `[N, M]` array as the augular fingerprints.

  """
  natoms = len(r)
  params = np.array(list(product(etas, gammas, zetas)))
  ndim = len(params)
  rr = r + np.eye(natoms) * rc
  r2 = rr ** 2
  rc2 = rc ** 2
  fr = cutoff_fxn(rr, rc)
  x = np.zeros((natoms, ndim))

  for l, (eta, gamma, zeta) in enumerate(params):
    for i in range(natoms):
      for j in range(natoms):
        if j == i:
          continue
        for k in range(natoms):
          if k == i or k == j:
            continue
          rij = coords[j] - coords[i]
          rik = coords[k] - coords[i]
          theta = np.dot(rij, rik) / (r[i, j] * r[i, k])
          v = (1 + gamma * theta)**zeta
          v *= np.exp(-eta * (r2[i, j] + r2[i, k] + r2[j, k]) / rc2)
          v *= fr[i, j] * fr[j, k] * fr[i, k]
          x[i, l] += v
    x[:, l] *= 2.0 **(1 - zeta)

  return x / 2.0


def naive_test():
  from pymatgen.io.xyz import XYZ
  from os.path import join

  xyzfile = join("..", "datasets", "B20pbe_opted.xyz")
  cart_coords = XYZ.from_file(xyzfile).molecule.cart_coords
  rc = 6.5
  fingers = get_behler_fingerprints(cart_coords, rc)
  v = [
    7.4962211769419, 3.908699186062, 0.9606682539908, 0.02123383466,
    25.00382305735, 15.2551697332, 6.6248599213, 0.6102237982
  ]
  diff = np.linalg.norm(fingers[0] - v)
  assert diff < 1e-6


if __name__ == "__main__":
  naive_test()
