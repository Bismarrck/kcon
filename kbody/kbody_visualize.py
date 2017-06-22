# coding=utf-8
"""
Plot the linear regression result of predicted energies and real energies.
"""

from __future__ import print_function, absolute_import

import numpy as np
import seaborn as sns
from constants import pyykko
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from os.path import isfile
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


eval_file = "./eval.npz"


def plot_reg():
  """
  Plot the predicted and real energies from the file 'eval.npz'.
  """
  if not isfile(eval_file):
    print("The evaluation result, 'eval.npz', cannot be found!")
  else:
    ar = np.load(eval_file)
    y_true, y_pred = np.negative(ar["y_true"]), np.negative(ar["y_pred"])
    y_diff = y_true - y_pred

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std = float(np.std(y_diff))
    score = r2_score(y_true, y_pred)
    num_evals = len(y_pred)

    plt.title("N=%d, MAE=%.3f eV, RMSE=%.3f eV, Var=%.3f eV, $R^2$=%.3f" % (
      num_evals, mae, rmse, std, score), fontsize=16)
    plt.xlabel("Real Energy (eV)", fontsize=14)
    plt.ylabel("Predicted Energy (eV)", fontsize=14)
    sns.regplot(x=y_true, y=y_pred)
    plt.show()


_RdBu_data = (
  (0.40392156862745099, 0.0, 0.12156862745098039),
  (0.69803921568627447, 0.09411764705882353, 0.16862745098039217),
  (0.83921568627450982, 0.37647058823529411, 0.30196078431372547),
  (0.95686274509803926, 0.6470588235294118, 0.50980392156862742),
  (0.99215686274509807, 0.85882352941176465, 0.7803921568627451),
  (0.96862745098039216, 0.96862745098039216, 0.96862745098039216),
  (0.81960784313725488, 0.89803921568627454, 0.94117647058823528),
  (0.5725490196078431, 0.77254901960784317, 0.87058823529411766),
  (0.2627450980392157, 0.57647058823529407, 0.76470588235294112),
  (0.12941176470588237, 0.4, 0.67450980392156867),
  (0.0196078431372549, 0.18823529411764706, 0.38039215686274508)
)

_BuRd_data = list(reversed(_RdBu_data))

# The colormap of `BuRd` which is just the reverse of `RdBu`.
cm = LinearSegmentedColormap.from_list("BuRd", _BuRd_data)


class _TruncatedGauss2D:

  def __init__(self, x, y, sigma):
    self._x = x
    self._y = y
    self._sigma = sigma
    self._scale = 1.0 / sigma / (np.sqrt(2.0 * np.pi))
    self._s2 = 2.0 * sigma ** 2
    self._thres = self._s2 * 9

  def __call__(self, x, y):
    d2 = (x - self._x) ** 2 + (y - self._y) ** 2
    if d2 > self._thres:
      return 0.0
    else:
      return np.exp(-d2 / self._s2) * self._scale


def _round_to_five(v):
  """
  Round the values to its nearest `*.5`.
  """
  return np.round(v * 2.0) / 2.0


def _get_bounds(xy):
  """
  Return the boundary of the coordinates.
  """
  minima = _round_to_five(xy.min(axis=0)) * 1.5
  maxima = _round_to_five(xy.max(axis=0)) * 1.5
  return minima.tolist() + maxima.tolist()


def _get_bond_vertices(x, y, i, j):
  """
  Return the vertices of the bond between atom i and j.
  """
  xi, yi = x[i], y[i]
  xj, yj = x[j], y[j]
  dx = xj - xi
  dy = yj - yi
  xi, xj = xi + dx * 0.25, xj - dx * 0.25
  yi, yj = yi + dy * 0.25, yj - dy * 0.25
  return [xi, xj], [yi, yj]


def colorbar(fig, ax, pcolor, label_values=True):
  """
  Add a color to the figure.
  
  Args:
    fig: a figure.
    ax: a `matplotlib.axes.Axes`.
    pcolor: a `matplotlib.axes.Axes.pcolor`.
    label_values: a boolean indicating whether use values or texts to describe 
      the colorbar.

  """
  cbar = fig.colorbar(pcolor, ax=ax)
  if label_values:
    texts = [s.get_text() for s in cbar.ax.get_yticklabels()]
    yticks = [""] + ["{} eV".format(x) for x in texts[1:-1]] + [""]
  else:
    yticks = [""] * len(cbar.ax.get_yticklabels())
    n = len(yticks)
    if n % 2 == 1:
      yticks[1] = "Stable"
      yticks[n // 2] = "Standard"
      yticks[-2] = "Unstable"
  cbar.ax.tick_params(labelsize=14)
  cbar.ax.set_yticklabels(yticks)


def atomic_heatmap2d(species, coordinates, ediff, ax):
  """
  Plot the atomic heatmap of a 2D molecule.
  
  Args:
    species: a `List[str]` as the atomic species.
    coordinates: a 2D array of shape [N, 3] as the atomic coordinates.
    ediff: a 1D array of shape [N, ] as the relative atomic energies.
    ax: a `matplotlib.axes.Axes` for plotting the heatmap.

  Returns:
    pcolor: a `matplotlib.axes.Axes.pcolor` for rendering the heatmap.

  """
  assert len(coordinates.shape) == 2
  natoms = len(species)

  # Center the coordinates
  xy = coordinates[:, :2]
  xy -= np.mean(xy, axis=0)
  x, y = xy[:, 0], xy[:, 1]

  # Generate the meshgrid
  bounds = _get_bounds(xy)
  nxp = int((bounds[2] - bounds[0]) / 0.05) + 1
  nyp = int((bounds[3] - bounds[1]) / 0.05) + 1
  lx = np.linspace(bounds[0], bounds[2], num=nxp, endpoint=True)
  ly = np.linspace(bounds[1], bounds[3], num=nyp, endpoint=True)
  xx, yy = np.meshgrid(lx, ly)
  zz = np.zeros_like(xx)

  # Compute the z values
  sigmas = np.array([pyykko[specie] for specie in species])
  gauss = [_TruncatedGauss2D(x[i], y[i], sigmas[i]) for i in range(natoms)]
  for i, j in np.ndindex(xx.shape):
    px, py = xx[i, j], yy[i, j]
    for k in range(natoms):
      zz[i, j] += ediff[k] * gauss[k](px, py)
  for i, j in np.ndindex(zz.shape):
    if -0.03 < zz[i, j] < 0.03:
      zz[i, j] = 0.0

  # Render the heatmap
  pcolor = ax.pcolor(xx, yy, zz, cmap=cm, vmin=-1.5, vmax=1.5)
  kwargs = dict(fontsize=14, horizontalalignment='center',
                verticalalignment='center')
  ax.tick_params(labelbottom='off', labelleft='off')

  # Render the molecule skeleton
  for k in range(natoms):
    ax.text(x[k], y[k], species[k], **kwargs)
  r = np.array([pyykko[specie] for specie in species])
  for i in range(natoms):
    for j in range(i + 1, natoms):
      dij = np.linalg.norm(xy[i] - xy[j])
      if dij <= 1.5 * (r[i] + r[j]):
        bx, by = _get_bond_vertices(x, y, i, j)
        ax.plot(bx, by, "-", color='black')

  return pcolor


if __name__ == "__main__":
  plot_reg()
