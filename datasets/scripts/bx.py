from __future__ import print_function

import numpy as np
import re
import time
import sys
from os.path import join


clusters = {"B28-": 28, "B35-": 35, "B37-": 37, "B38-": 38, "B39-": 39}


def get_regex_patterns(opted):
  if not opted:
    energy_patt = re.compile(r"i\s=\s+\d+,\sE\s=\s+([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
  else:
    energy_patt = re.compile(r"([\w.-]+)")
    string_patt = re.compile(
      r"([A-Za-z]+)\s+([\w.-]+)\s+([\w.-]+)\s+([\w.-]+)")
  return energy_patt, string_patt


def get_xyzfile(cluster, opted=False):
  if opted:
    return "{}_opted.xyz".format(cluster)
  else:
    return "{}.xyz".format(cluster)


def extract_xyz(filename, num_atoms, opted=False, verbose=False):
  num_examples = 1000
  energies = np.zeros((num_examples,))
  coords = np.zeros((num_examples, num_atoms, 3))
  array_of_species = []
  species = []
  stage = 0
  i = 0
  j = 0
  n = None
  ener_patt, xyz_patt, = get_regex_patterns(opted)

  tic = time.time()
  if verbose:
    sys.stdout.write("Extract cartesian coordinates ...\n")

  with open(filename) as f:
    for line in f:
      if i == num_examples:
        num_examples *= 2
        energies = np.resize(energies, (num_examples, ))
        coords = np.resize(coords, (num_examples, num_atoms, 3))
      l = line.strip()
      if l == "":
        continue
      if stage == 0:
        if l.isdigit():
          n = int(l)
          if n > num_atoms:
            raise ValueError("The number of atoms %d from the file is larger "
                             "than the given maximum %d!" % (n, num_atoms))
          stage += 1
      elif stage == 1:
        m = ener_patt.search(l)
        if m:
          energies[i] = float(m.group(1))
          stage += 1
      elif stage == 2:
        m = xyz_patt.search(l)
        if m:
          coords[i, j, :] = [float(v) for v in m.groups()[1:4]]
          species.append(m.group(1))
          j += 1
          if j == n:
            array_of_species.append(species)
            species = []
            j = 0
            stage = 0
            i += 1
            if verbose and i % 1000 == 0:
              sys.stdout.write("\rProgress: %7d  /  %7d" % (i, num_examples))
    if verbose:
      print("")
      print("Total time: %.3f s\n" % (time.time() - tic))

  if i < num_examples:
    array_of_species = np.asarray(array_of_species)
    energies = np.resize(energies, (i, ))
    coords = np.resize(coords, (i, num_atoms, 3))

  return array_of_species, energies, coords


def skewness(vector):
  v = np.asarray(vector)
  sigma = np.std(v)
  s = np.mean((v - v.mean())**3.0)
  eps = 1E-8
  if np.abs(sigma) < eps or np.abs(s) < eps:
    return 0.0
  else:
    return s / (sigma**3.0)


def get_usr_features(cart_coords):
  """
  Return the USR feature vector.
  """
  def get_vector(v1, v2, v3, v4, coords):
    vector = np.zeros(12)
    k = 0
    for v in [v1, v2, v3, v4]:
      di = np.linalg.norm(v - coords, axis=1)
      vector[k: k+3] = np.mean(di), np.std(di), skewness(di)
      k += 3
    return vector

  x = cart_coords.mean(axis=0)
  d = np.linalg.norm(x - cart_coords, axis=1)
  y = cart_coords[np.argmin(d)]
  z = cart_coords[np.argmax(d)]
  d = np.linalg.norm(z - cart_coords, axis=1)
  w = cart_coords[np.argmax(d)]

  return get_vector(x, y, z, w, cart_coords)


def get_all_fingerprints(coords):
  ntotal = coords.shape[0]
  fingerprints = np.zeros((ntotal, 12))
  for i in range(ntotal):
    fingerprints[i] = get_usr_features(coords[i])
  return fingerprints


def remove_duplicates(energies, coords, threshold=0.97):
  keep = np.ones_like(energies, dtype=bool)
  ntotal = len(energies)
  fingerprints = get_all_fingerprints(coords)
  for i in range(1, ntotal):
    ediff = np.abs(energies[i] - energies[:i])
    indices = np.where(ediff < 0.02)[0]
    if len(indices) == 0:
      continue
    v = np.sum(np.abs(fingerprints[i] - fingerprints[indices]), axis=1)
    sims = 1.0 / (1.0 + v) 
    if sims.max() > threshold:
      keep[i] = False
  return energies[keep], coords[keep]


def to_xyz_strings(species, energy, coords):
  fmt = "{:2s}  {: 14.8f} {: 14.8f} {: 14.8f}"
  n = len(species)
  strings = ["{:d}".format(n), "{:.8f}".format(energy)]
  strings.extend([fmt.format(species[i], *coords[i]) for i in range(n)])
  return strings


with open("Bx-.xyz", "w+") as f:

  for cluster, num_atoms in clusters.items():

    print("Processing {} ...".format(cluster))

    opted = False
    xyzfile = get_xyzfile(cluster, opted=opted)
    array_of_species, energies, coords = extract_xyz(
      xyzfile, num_atoms=num_atoms, opted=opted, verbose=False
    )

    print("Total {:4d} isomers parsed from {}.".format(len(energies), xyzfile))
    print("Energy Range: {: 8.4f} a.u - {: 8.4f} a.u".format(energies.max(), energies.min()))

    thres = 0.97
    energies, coords = remove_duplicates(energies, coords, threshold=thres)
    n = len(energies)
    array_of_species = array_of_species[:n]

    print("Total {:4d} unique isomers with threshold {:.3f}".format(n, thres))
    print("Energy Range: {: 8.4f} a.u - {: 8.4f} a.u".format(energies.max(), energies.min()))
    print("")

    lines = []
    for i in range(n):
      lines.extend(to_xyz_strings(array_of_species[i], energies[i], coords[i]))
    f.write("\n".join(lines))
    f.flush()

