## SCoNE

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Input](#input)
- [Output](#output)
- [Instructions for use](#instructions-for-use)
- [Unsupervised phenotyping algorithms](#unsupervised-phenotyping-algorithms)
- [Additional Source Code](#additional-source-code)
- [Other notes](#other-notes)
- [Common pitfalls](#common-pitfalls)

## Overview
This README accompanies the paper 'Sparse Covariate-aware Non-negative Extraction Improves Complex Disease Subtyping'. The contribution of the paper is to introduce a novel unsupervised phenotyping method for integrating clinical and genetic data that is scalable, able to handle count-based and continuous data, and accounts for sparsity and unwanted sources of variation in the data.

## System Requirements

### Hardware Requirements
The code runs on a standard computer. Memory requirements depend on the size of the input data, and larger datasets may require additional RAM.

### Software Requirements

#### OS Requirements
The code has been developed and tested on Ubuntu 24.04. It is expected to run on other modern Linux distributions.

#### Python Dependencies
SCoNE depends on the packages listed in `requirements.txt`:

```text
Python 3.10+
numpy
scipy
scikit-learn
joblib
pandas
```

We recommend installing these dependencies in a new Python virtual environment (e.g., using `venv` or Conda) to avoid conflicts with existing packages.


Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

Optional: To run SCoNE on GPU, install the appropriate version of CuPY for your CUDA installation. An example is provided in `requirements-gpu.txt`.

#### Optional R dependencies (for running comparator methods)
To use the provided wrappers for comparator methods, the following R packages must be installed:

```text
rgwas
data.table
jsonlite
mvcluster
```

The `rgwas` and `mvcluster` packages are not available on CRAN and should be installed according to the installation instructions provided by its authors.

## Installation Guide

### Install from Github

```bash
git clone https://github.com/G2Lab/SCoNE.git
cd SCoNE
pip install -r requirements.txt
```

Typical install time: Less than 5 minutes on a standard desktop computer.

## Demo

A complete demonstration of SCoNE and all comparator methods is provided in the Jupyter notebook `code/demo.ipynb`. The notebook reproduces the example workflow from data loading through model fitting and evaluation.

After loading the input matrices (`G`, `C`, and `Z`), SCoNE can be run with a single function call:


```python
import pandas as pd
import os
from algorithms.SCoNE import SCoNE_parallel

cwd = os.getcwd()
parent_dir = os.path.dirname(os.getcwd())

G = pd.read_csv(f'{parent_dir}/example_data/G.csv', index_col=0).to_numpy(dtype=float)
C = pd.read_csv(f'{parent_dir}/example_data/C.csv', index_col=0).to_numpy(dtype=float)
Z = pd.read_csv(f'{parent_dir}/example_data/Z.csv', index_col=0).to_numpy(dtype=float)

factor_matrices, loss_function = SCoNE_parallel(
    G, 
    C, 
    Z, 
    rank=3,
    alpha=max(G.max(),C.max())**2,
    lambda_H_G=1e-4, 
    lambda_H_C=1e-4, 
    lambda_Gloss=1,  
    num_init=3,
    init='nndsvda',
    G_loss_type='kl_div', 
    C_loss_type='kl_div', 
)
```

Typical run time: Approximately 20 seconds on a standard desktop computer.

## Input

SCoNE takes three NumPy arrays (dtype=float) together with the desired decomposition rank:

- **G** (`N × M_G`): Genetic data matrix, where rows correspond to samples and columns correspond to genetic features.
- **C** (`N × M_C`): Clinical data matrix, where rows correspond to the same samples and columns correspond to clinical features.
- **Z** (`N × M_Z`): Covariate matrix containing an intercept column and sample-level covariates (e.g., age, sex, admixture proportions).

All matrices must have the same number of rows (`N`), corresponding to the same set of samples.

## Output
SCoNE returns two objects:

- **factor_matrices**: A dictionary containing the learned factor matrices.
    - **W** (`N × rank`): participant factor matrix
    - **H_G** (`M_G × rank`): genetic factor matrix
    - **H_C** (`M_C × rank`): clinical factor matrix
    - **U_G** (`M_G × M_Z`): covariate effects matrix for genetic decomposition
    - **U_C** (`M_C × M_Z`): covariate effects matrix for clinical decomposition
  
- **loss_function**: A dictionary containing the total loss and the contribution of each loss component at every optimization iteration. It also records the norms and sparsity of the factor matrices throughout optimization, as well as the final cophenetic correlation coefficient. 


## Instructions for use
The primary entry point to SCoNE is the `SCoNE_parallel` function in `code/algorithms/SCoNE.py`. The required inputs are the genetic (`G`), clinical (`C`), and covariate (`Z`) matrices together with the desired decomposition rank.

The remaining arguments control initialization, regularization, and optimization settings. The example in `code/demo.ipynb` demonstrates a typical configuration.

For a complete description of all function arguments, refer to the documentation in the `SCoNE_parallel` docstring.

## Unsupervised phenotyping algorithms
The table below lists all unsupervised phenotyping algorithms applied in the paper, whether they are considered baseline or comparator methods, and the python file with their implementation (in 'code/algorithms'). 

| Algorithm  | Type         | Python file   | 
| ---------- | ------------ | ------------- | 
| SCoNE      |              | SCoNE         | 
| HNMF       | Comparator   | SCoNE         | 
| HNMF(res)  | Baseline     | SCoNE         | 
| C-CoNE     | Baseline     | SCoNE         | 
| G-CoNE     | Baseline     | SCoNE         | 
| C-NMF      | Baseline     | SCoNE         | 
| G-NMF      | Baseline     | SCoNE         | 
| RGWAS      | Comparator   | RGWASWrapper  | 
| MVBC       | Comparator   | MVBCWrapper   | 

Refer to 'code/demo.ipynb' for a tutorial on the implementations of all methods. We simulate G, C, and Z using simulate_views() function in 'code/simulate_data.py' for 1000 samples, 10 clinical features, 10 genetic features, and 3 covariates (plus intercept) with true low rank 3. 
All methods will return two dictionaries: one containing learned factor matrices and the other containing recorded loss. 

## Additional Source Code

The following files are included to support transparency and reproducibility but are not required to run the SCoNE algorithm or reproduce the demonstration in `demo.ipynb`.

### Simulated Data

The script `code/test_reconstruction.py` contains the code used to simulate datasets and evaluate SCoNE and the comparator methods described in the manuscript. The notebook `code/test_reconstruction.ipynb` contains the code used to visualize and analyze these results.

### Algorithm Evaluation

Utilities used to evaluate algorithm performance on simulated data are provided in `code/evaluation`, primarily in `reconstruction_evaluation.py`.

### Application to All of Us

The code used to apply SCoNE to the All of Us Researcher Workbench (v1.0) is available in the companion repository:

`https://github.com/G2Lab/allofus_unsupervised_pheno`

## Other notes
The SCoNE_parallel() function has n_jobs=1 as default, meaning the initializations run sequentially rather than in parallel. This is often preferable for large problems because it allows NumPy’s internal multithreading to fully utilize available CPU cores for each run. For smaller problems, increasing n_jobs can improve performance by running multiple initializations in parallel.

## Common pitfalls
Here are a few additional setup instructions to keep in mind:
1. Make sure that the features of Z are generally on the same scale. For example, if Z consists of admixture fractions, sex at birth (binary), and year of birth, make sure to min-max normalize year of birth so it sits on the same 0-1 scale as other variables.
2. If population structure features are admixture fractions, drop one to avoid perfect multicollienarity of features. 
3. Z should always have an intercept term.
3. For initialization, we use sklearns initialize_nmf. If the input matrices C and G are not represented as floats before input and instead are integers, sklearn random initialization will cause extreme sparsity in the initial factor matrices, which will cause poor decomposition results. The simulate_data function we use in the demo returns .astype(float).