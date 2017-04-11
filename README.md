# Grendel

Deep Learning on Chemistry

* Author: Xin Chen
* Email: Bismarrck@me.com

### Introduction

There are three mini-projects:

1. mbenn:
  * A TensorFlow implementation of Alexandrova's MBE-NN-M network with B20 as the testing cluster.
2. behler:
  * A TensorFlow convolution based implementation of Behler's atomic neural network with B20 as the testing cluster.
3. kbody:
  * The main part of this project.
  * A TensorFlow implementation of the deep convolution network named `sum-kbody-cnn`.
  * This model has combined the advantages of Alexandrova's MBE-NN-M and Behler's ANN.

### Requirements

All codes are written with Python3.6 standard. Some codes can be executed perfectly under Python2.7.x.

1. tensorflow>=1.0
2. numpy
3. scipy
4. jupyter
5. matplotlib
6. scikit-learn
