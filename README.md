# kCON

kCON is a scalable and transferable deep learning framework for chemistry with the ability to provide insight into atomistic structures of varying stoichiometry from small and scrap training sets.

* Author: Xin Chen
* Email: Bismarrck@me.com

![kCON][image-1]

### 1. Related work

* Exploring atomic chemistry with machine learning

### 2. Requirements

All codes are written in Python3.6. Some codes can be executed perfectly under Python2.7.x.

1. tensorflow\>=1.3
2. numpy
3. scipy
4. jupyter
5. matplotlib
6. scikit-learn
7. ase\>=3.12

### 3. Datasets

In order to pull these dataset files from github, **[Git LFS][1]** should be installed.

There some built-in dataset, including:

- QM7
- GDB-9
- Napthalene20k
- Quinoline (PBE, DFTB)
- Anatase Titanium Dioxide (DFTB)

### 4. Modules

This project is organized like other projects under [tensorflow/models][2].

#### Model

1. `inference.py`: for the detailed implementation of the **kCON** model.
2. `kcnn.py`: for constructing **kCON** model and building the loss function for CPU/single-GPU training.
3. `transformer.py`: for transforming [`ase.Atoms`][3] to input features of the model.
4. `save_model.py`: for freezing and exporting trained model to `pb` files.

#### Training and validation

1. `train.py`: the CPU/single-GPU version of training.
2. `multi_gpu_train.py`: for training on a single node with multiple GPUs.
3. `distributed_train.py`:  for training on distributed systems. **Not implemented yet**.
4. `evaludation.py`: for continuously evaluating the training performance.

#### I/O

1. `database.py`: for parsing xyz files and managing the sqlite database files generated by [`ase.db`][4].
2. `pipeline.py`: for reading [TensorFlow Records][5] files to build input pipeline for training and continuous evaluation. `tf.contrib.data.Dataset` API is utilized.
3. `build_dataset.py`: for transforming raw xyz files to ASE databases and TFRecords.

#### Prediction

1. `predictor.py`: for making predictions using trained models.
	- **Currently broken**
2. `calculator.py`: an [`ase.calculator.Calculator`][6] wrapper of `KcnnPredictor`.
	- **Currently broken**

#### Auxiliary modules

1. `constants.py`: global constants are defined in this module.
2. `utils.py`: some helper functions are defined in this module.

### 5. Visualization

One of the advantage of using convolutional neural network in chemistry is that we can visualize how the network learns chemical patterns. Chemical patterns are somehow very similar to the low-level image features like lines or circles. The traditional CNN visualization methods can be applied directly. Here is a demo of kCON applying on the quinoline PBE dataset.

![Visualization][image-2]

### 6. Atomic Energy

The concept of artificial neural network derived atomic contributions to total energies was first proposed in 2007 by [Behler et al][7]. Despite that many later machine learning models include the concept of atomic energy, very little work has been done on interpreting the chemical meaning of these contributions and utilizing them in chemical applications.  We did some qualitative and statistic analysis of the atomic energies learned from kCON and successfully proved that these atomic energies can perfectly agree with our chemical intuitions from valence-bond theory, thus providing us with a new approach to understand the local stabilities of atoms and molecular moieties. 

Here is an example of using atomic energies to study the stability of molecules.

![Stability][image-3]

**a)** Configurational DFT and Neural Network (NN) energy spectra of Quinoline within 3.5 eV of the global minimum. The DFT energy of the global minimum is the reference energy. Isomers labeled red are included in the test dataset. **i)** and **j)** are energies of C and H atoms in the global minimum, and their averages are used as reference for other atomic energies, as shown in **h)**. **b)**, **c)**, **d)**, **e)**, **f)** and **g)** are examples of analysis based on relative atomic energies. 

[1]:	https://git-lfs.github.com
[2]:	https://github.com/tensorflow/models
[3]:	https://wiki.fysik.dtu.dk/ase/ase/atoms.html
[4]:	https://wiki.fysik.dtu.dk/ase/ase/db/db.html#ase-db
[5]:	https://www.tensorflow.org/versions/r1.1/programmers_guide/reading_data
[6]:	https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
[7]:	https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401

[image-1]:	doc/images/figure1.png
[image-2]:	./doc/images/figure2.png
[image-3]:	doc/images/figure4.png