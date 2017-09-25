# The Guide of kCON

* Author: Xin Chen
* Email: Bismarrck@me.com

## 1. Dataset Format

Currently two types of **XYZ** files are supported.
* **xyz**: the default XYZ format. Atomic forces are not included.
* **ase**: an extension of the origin XYZ format. Atomic forces, periodic boundary conditions and cell parameters are all included.

### a) xyz

```
5
-0.66606164
C      0.99825996    -0.00246000    -0.00436000
H      2.09020996    -0.00243000     0.00414000
H      0.63378996     1.02686000     0.00414000
H      0.62703997    -0.52772999     0.87810999
H      0.64135998    -0.50746995    -0.90539992
```

These default datasets are in **xyz** format:

1. C9H7N.PBE
2. C9H7Nv1
3. TiO2
4. qm7
5. gdb9
6. Bx-

### b) ase

```
9
Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:Z:I:1:magmoms:R:1:tags:I:1:forces:R:3 energy=-4214.522353757427 pbc="F F F"
C     -0.26468800    -0.43877700    -0.09813700   6    0.00000000        0    -0.26388656    -1.00787160     1.26421434
C     -1.02913600     0.78256100     0.47120800   6    0.00000000        0     3.28335744     1.63116404     1.08721937
O      1.16243900    -0.36308100    -0.32648800   8    0.00000000        0     0.23767131     1.21706570    -0.41894319
H     -0.41283400    -1.32819100     0.55837500   1    0.00000000        0     0.66553025     0.54817454     0.48471736
H     -0.76215900    -0.88657200    -0.93062500   1    0.00000000        0    -0.52741611     0.31781506    -1.63151403
H     -1.63563900     0.71746900     1.38827900   1    0.00000000        0    -0.23608035    -0.85857582    -0.35843090
H     -0.11712300     1.46064600     0.95972600   1    0.00000000        0    -2.09373097    -1.01380144    -1.53418677
H     -1.58244400     1.30150800    -0.31839200   1    0.00000000        0     0.13799709     0.38674910     0.13623815
H      1.47470100     0.40039500    -0.92044900   1    0.00000000        0    -1.20122614    -1.21905564     0.96704276
```

These default datasets are in **ase** format:

1. naphthalene20k
2. md3k
3. ethanol10k

### c) How to build a dataset

To build a dataset, we can run the following command:

```bash
python -u build_dataset.py --dataset=qm7 --num_examples=7165 --format=xyz
```

or

```bash
python -u build_dataset.py --dataset=ethanol10k --forces --num_examples=10000 \
       --format=ase
```

## 2. Training

Requirements:

1. Python\>=3.6.0
2. ase\>=3.12
3. TensorFlow\>=1.3.0
4. numpy\>=1.12.0
5. scipy\>=0.19.0
6. scikit-learn\>=0.19.0

After building a dataset, we can run the following command to start training a 
**kCON** model:

```bash
python -u train.py --dataset=qm7 --num_epochs=1000 --log_frequency=5 \
       --conv_sizes=40,60,60,40 --learning_rate=0.0004 --train_dir=qm7_lr4
```

or

```bash
python -u train.py --dataset=ethanol10k --num_epochs=1000 --log_frequency=5 \
       --conv_sizes=40,60,60,40 --learning_rate=0.0001 --forces \
       --normalizer=layer_norm --train_dir=ethanol_lr1_ln
```

The training may take a few or tens of hours. So if the node can't access to a 
batch system (PEB, Slurm), one can use `nohup` to do background training:

```bash
nohup python -u train.py --dataset=ethanol10k --num_epochs=1000 \
             --log_frequency=5 --conv_sizes=40,60,60,40 --learning_rate=0.0001 \
             --forces --normalizer=layer_norm \
             --train_dir=ethanol_lr1_ln &> ethanol.log &
```