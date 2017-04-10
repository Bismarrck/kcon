# coding=utf-8
"""
Plot the linear regression result of predicted energies and real energies.
"""

from __future__ import print_function, absolute_import

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
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


if __name__ == "__main__":
  plot_reg()
