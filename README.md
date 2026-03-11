## Overview
This README accompanies the paper 'Sparse Covariate-aware Non-negative Extraction Improves Complex Disease Subtyping'. The contribution of the paper is to introduce a novel unsupervised phenotyping method for integrating clinical and genetic data that is scalable, able to handle count-based and continuous data, and accounts for sparsity and unwanted sources of variation in the data.

## Unsupervised phenotyping algorithms
The table below lists all unsupervised phenotyping algorithms applied in the paper, whether they are considered baseline or comparator methods, and the python file with their implementation (in 'code/algorithms'). The following sections provide more detail on the implementation of each algorithm.

| Algorithm  | Type         | Python file   | 
| ---------- | ------------ | ------------- | 
| SCoNE      |              | SCoNE         | 
| SCoNE(Fro) | Baseline     | SCoNE         | 
| CoNE       | Baseline     | SCoNE         | 
| HNMF       | Comparator   | SCoNE         | 
| HNMF(res)  | Baseline     | SCoNE         | 
| C-CoNE     | Baseline     | SCoNE         | 
| G-CoNE     | Baseline     | SCoNE         | 
| C-NMF      | Baseline     | SCoNE         | 
| G-NMF      | Baseline     | SCoNE         | 
| RGWAS      | Comparator   | MVBCWrapper   | 
| MVBC       | Comparator   | RGWASWrapper  | 

We will illustrate the implementation of each of these algorithms using simulated data. For ease of use, this demo is also in 'code/demo.ipynb'. We simulate G, C, and Z using simulate_views() function in 'code/simulate_data.py' for 100 samples, 10 clinical features, 10 genetic features, and 5 covariates with true low rank 3. To do so, run the following code:
```python
# import necessary libraries
import numpy as np
from simulate_data import simulate_views
matrices = simulate_views(n=100, num_genes=10, M_C=10, M_Z=5, rank=3, seed=0)
# drop multicollinear column of Z and add intercept to Z
matrices['Z'] = np.c_[np.ones(matrices['Z'][:,1:].shape[0]), matrices['Z'][:,1:]]
```

All methods will return two dictionaries: one containing learned factor matrices and the other containing recorded loss.

### SCoNE
All methods that use the python file SCoNE for their implementation take matrices as input in numpy form. We provide the function with required parameters G, C and Z, as well as the rank of the decomposition. Other optional parameters we include are regularization parameters alpha, lambda_H_G, and lambda_H_C. We set num_init=10, which represents the number of initializations to run. Initializations can be random (init='random'), or variations of NNSVD ('nndsvd', 'nndsvda', 'nndsvdar'). We set lambda_Gloss=1 indicating that the views have equal relative weight. We set the loss for G and C decomposition to both be KL-divergence. There are additional parameters such as tolerance and min/max iterations but we leave them as default. Refer to SCoNE.py for more information on these.

```python
from algorithms.SCoNE import SCoNE_parallel
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],matrices['C'],matrices['Z'], rank=3,
    alpha=max(matrices['G'].max(),matrices['C'].max())**2,lambda_H_G=1e-4, lambda_H_C=1e-4, lambda_Gloss=1,   # regularization parameters
    num_init=10,init='nndsvda',G_loss_type='kl_div', C_loss_type='kl_div', # in 'kl_div',  'fro' or None
)
```

### SCoNE (Fro)
SCoNE (Fro) represents SCoNE performed assuming a Frobenius loss instead of KL divergence. Therefore, we only change G_loss_type and C_loss_type.

```python
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],matrices['C'],matrices['Z'], rank=3,
    alpha=max(matrices['G'].max(),matrices['C'].max())**2,lambda_H_G=1e-4, lambda_H_C=1e-4, lambda_Gloss=1,         # regularization parameters
    num_init=10,init='nndsvda',G_loss_type='fro', C_loss_type='fro', # in 'kl_div',  'fro' or None
)
```

### CoNE

CoNE represents SCoNE with no regularization, so we set alpha, lambda_H_G and lambda_H_C to zero.
```python
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],matrices['C'],matrices['Z'], rank=3,
    alpha=0,lambda_H_G=0, lambda_H_C=0, lambda_Gloss=1,         # regularization parameters
    num_init=10,init='nndsvda',G_loss_type='kl_div', C_loss_type='kl_div', # in 'kl_div',  'fro' or None
)
```

### HNMF

HNMF represents SCoNE with no regularization and no removal of unwanted variation represented in covariate matrix Z. We set alpha, lambda_H_G and lambda_H_C to zero and set Z to None.
```python
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],matrices['C'],None, rank=3,
    alpha=0,lambda_H_G=0, lambda_H_C=0, lambda_Gloss=1,         # regularization parameters
    num_init=10,init='nndsvda',G_loss_type='kl_div', C_loss_type='kl_div', # in 'kl_div',  'fro' or None
)
```
### HNMF (res)
HNMF (res) simply represents running HNMF on matrices C and G after residualizing out Z. We set the loss for both to be Frobenius loss since after residualization the data no longer represents counts. The code can look something like this:
```python
from algorithms.SCoNE import proj_nonneg

C_resid = proj_nonneg(matrices['C'] - matrices['Z'] @ np.linalg.lstsq(matrices['Z'], matrices['C'],rcond=None)[0])
G_resid = proj_nonneg(matrices['G'] - matrices['Z'] @ np.linalg.lstsq(matrices['Z'], matrices['G'],rcond=None)[0])

factor_matrices, loss_function = SCoNE_parallel(
    G_resid,C_resid,None, rank=3,
    alpha=0,lambda_H_G=0, lambda_H_C=0, lambda_Gloss=1,         # regularization parameters
    num_init=10,init='nndsvda',G_loss_type='fro', C_loss_type='fro', # in 'kl_div',  'fro' or None
)
```

### C-CoNE
C-CoNE performs a covariate-aware decomposition of the C matrix only. To remove consideration of G, we set G=None and G_loss_type as None.
```python
factor_matrices, loss_function = SCoNE_parallel(
    None,matrices['C'],matrices['Z'], rank=3,
    num_init=10,init='nndsvda',G_loss_type=None, C_loss_type='kl_div', # in 'kl_div',  'fro' or None
)
```

### G-CoNE
G-CoNE performs a covariate-aware decomposition of the G matrix only. To remove consideration of C, we set C=None and C_loss_type as None.
```python
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],None,matrices['Z'], rank=3,
    num_init=10,init='nndsvda',G_loss_type='kl_div', C_loss_type=None, # in 'kl_div',  'fro' or None
)
```

### C-NMF
C-NMF performs NMF on the C matrix. To remove consideration of G and Z, we set G=None, Z=None and G_loss_type as None.
```python
factor_matrices, loss_function = SCoNE_parallel(
    None,matrices['C'],None, rank=3,
    num_init=10,init='nndsvda',G_loss_type=None, C_loss_type='kl_div', # in 'kl_div',  'fro' or None
)
```

### G-NMF
G-NMF performs NMF on the G matrix. To remove consideration of C and Z, we set C=None, Z=None and C_loss_type as None.
```python
factor_matrices, loss_function = SCoNE_parallel(
    matrices['G'],None,None, rank=3,
    num_init=10,init='nndsvda',G_loss_type='kl_div', C_loss_type=None, # in 'kl_div',  'fro' or None
)
```

### RGWAS
We implement a python wrapper around the R code provided in Dahl et al. "Reverse GWAS: Using genetics to identify and model phenotypic subtypes". r_path should be set to the path to your Rscript. To implement RGWAS, we need to write G, C and Z to numpy or csv (with first column as index name) files.

```python
import os
from algorithms.RGWASWrapper import RGWASWrapper

cwd = os.getcwd()
parent_dir = os.path.dirname(os.getcwd())
np.save(f'{parent_dir}/example_data/G',matrices["G"])
np.save(f'{parent_dir}/example_data/C',matrices["C"])
np.save(f'{parent_dir}/example_data/Z',matrices["Z"])

factor_matrices, loss_function = RGWASWrapper(
    r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', # REPLACE WITH CORRECT Rscript path
    G_path=f'{parent_dir}/example_data/G.npy', C_path=f'{parent_dir}/example_data/C.npy', 
    Z_path=f'{parent_dir}/example_data/Z.npy', 
    rank=3, num_init=1)
```

### MVBC
We implement a python wrapper around the R code provided in Sun et al. "Multi-view Biclustering for Genotype-Phenotype Association Studies of Complex Diseases". r_path should be set to the path to your Rscript. To implement MVBC, we need to write G, C and Z to numpy  or csv (with first column as index name) files.

```python
from algorithms.MVBCWrapper import MVBCWrapper

factor_matrices, loss_function = MVBCWrapper(
    G_path=f'{parent_dir}/example_data/G.npy', C_path=f'{parent_dir}/example_data/C.npy', 
    rank=3, lambda_W=0.1, lambda_H_G=0.1, lambda_H_C=0.1, 
    r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript')
```

## Simulated data 
The python file 'code/test_reconstruction.py' contains code used to simulate data and run each algorithm as described in the paper. The jupyter notebook 'test_reconstruction.ipynb' contains code to graph and analyze results.

## Algorithm evaluation 
Utilities for algorithm evaluation on simulated data are stored in the evaluation folder, most importantly under the reconstruction_evaluation.py file.

## Other notes
The SCoNE_parallel() function has n_jobs=1 as default, meaning the initializations run sequentially rather than in parallel. This is often preferable for large problems because it allows NumPy’s internal multithreading to fully utilize available CPU cores for each run. For smaller problems, increasing n_jobs can improve performance by running multiple initializations in parallel.

## Common pitfalls
Here are a few additional setup instructions to keep in mind:
1. Make sure that the features of Z are generally on the same scale. For example, if Z consists of admixture fractions, sex at birth (binary), and year of birth, make sure to min-max normalize year of birth so it sits on the same 0-1 scale as other variables.
2. If population structure features are admixture fractions, drop one to avoid perfect multicollienarity of features. However, when one is dropped, you then need to ensure that Z contains an intercept term (this restores the ability of the model to represent the dropped component).
3. For initialization, we use sklearns initialize_nmf. If the input matrices C and G are not represented as floats before input and instead are integers, sklearn random initialization will cause extreme sparsity in the initial factor matrices, which will cause poor decomposition results. The simulate_data function we use in the demo returns .astype(float).