# CODE-AE

## CODE-AE - CONTEXT-AWARE DECONFOUNDING AUTOENCODER
-----------------------------------------------------------------

## 1. Introduction
**CODE-AE** is a Python implementation of Context-aware Deconfounding Autoencoder (CODE-AE) that can extract common biological signals masked by context-specific patterns and confounding factors.

## Architecture
![architecture](./figs/architecture.png?raw=true)


## 3. Installation

**CODE-AE** depends on Numpy, SciPy, PyTorch (CUDA toolkit if use GPU), scikit-learn, pandas. 
You must have them installed before using **CODE-AE**.

The simple way to install them is using Docker: the Dockerfile is given within folder ``code/``

## 4. Usage

### 4.1. Data

Benchmark datasets available at Zenodo[http://doi.org/10.5281/zenodo.4477674]

### 4.2 Example 
#### 4.2.1 Pre-train encoders
```sh
    $ python pretrain_hyper_main
```
Arguments in this script:
* ``--method``:       method to be used for encoder pre-training
* ``--train``:        retrain the encoders
* ``--no-train``:     no re-training
* ``--norm``:        use L2 normalization for embedding
* ``--no-norm``:     don't use L2 normalization for embedding


