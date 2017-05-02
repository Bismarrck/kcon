# coding=utf-8
from __future__ import print_function, absolute_import

import numpy as np
from kbody_predict import CNNPredictor
from kbody_input import extract_xyz
from matplotlib import pyplot as plt

xyzfile = "../test_files/B39-_opted.xyz"
num_examples = 2000
num_atoms = 39

array_of_species, y_true, coords, _ = extract_xyz(
  xyzfile, num_examples, num_atoms, xyz_format='xyz'
)

steps = [55000, 162026, 689034]
y_pred = {}
y_max = y_true.max()
y_min = y_true.min()

for train_step in steps:

  model_path = "./models/Bx-.v4/model.ckpt-{:d}".format(train_step)
  clf = CNNPredictor(["B"], model_path)
  y_total, _, _ = clf.predict(array_of_species[0], coords[0:-1:10])
  y_pred[train_step] = y_total
  y_max = max(y_total.max(), y_max)
  y_min = min(y_total.min(), y_min)


legends = {}

y_true_ = y_true[0:-1:10]
sort = np.argsort(y_true_)

y_true_ = y_true_[sort]
y_55k_ = y_pred[steps[0]][sort]
y_160k_ = y_pred[steps[1]][sort]
y_680k_ = y_pred[steps[2]][sort]

print("MAE (eV)    at  55k: ", np.mean(np.abs(y_55k_ - y_true_)))
print("stddev (eV) at  55k: ", np.std(np.abs(y_55k_ - y_true_)))

print("")

print("MAE (eV)    at 160k: ", np.mean(np.abs(y_160k_ - y_true_)))
print("stddev (eV) at 160k: ", np.std(np.abs(y_160k_ - y_true_)))

print("")

print("MAE (eV)    at 680k: ", np.mean(np.abs(y_680k_ - y_true_)))
print("stddev (eV) at 680k: ", np.std(np.abs(y_680k_ - y_true_)))

l, = plt.plot(y_true_, "g.")
legends["true"] = l

l, = plt.plot(y_55k_, "r.")
legends["55k"] = l

l, = plt.plot(y_160k_, "b.")
legends["160k"] = l

l, = plt.plot(y_680k_, "c.")
legends["380k"] = l

plt.ylim([y_min - 5, y_max + 5])
plt.legend(list(legends.values()), list(legends.keys()))
plt.show()

