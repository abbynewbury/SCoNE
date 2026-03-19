## Overview
This README accompanies the paper 'Sparse Covariate-aware Non-negative Extraction Improves Complex Disease Subtyping'. The contribution of the paper is to introduce a novel unsupervised phenotyping method for integrating clinical and genetic data that is scalable, able to handle count-based and continuous data, and accounts for sparsity and unwanted sources of variation in the data.

## Unsupervised phenotyping algorithms
The table below lists all unsupervised phenotyping algorithms applied in the paper, whether they are considered baseline or comparator methods, and the python file with their implementation (in 'code/algorithms'). The following sections provide more detail on the implementation of each algorithm.

| Algorithm  | Type         | Python file   | 
| ---------- | ------------ | ------------- | 
| SCoNE      |              | SCoNE         | 
| SCoNE(Fro) | Baseline     | SCoNE         | 
| HNMF       | Comparator   | SCoNE         | 
| HNMF(res)  | Baseline     | SCoNE         | 
| C-CoNE     | Baseline     | SCoNE         | 
| G-CoNE     | Baseline     | SCoNE         | 
| C-NMF      | Baseline     | SCoNE         | 
| G-NMF      | Baseline     | SCoNE         | 
| RGWAS      | Comparator   | MVBCWrapper   | 
| MVBC       | Comparator   | RGWASWrapper  | 

We will illustrate the implementation of of SCoNE using simulated data. Refer to 'code/demo.ipynb' for implementations of all methods. We simulate G, C, and Z using simulate_views() function in 'code/simulate_data.py' for 1000 samples, 10 clinical features, 10 genetic features, and 3 covariates (plus intercept) with true low rank 3. 
All methods will return two dictionaries: one containing learned factor matrices and the other containing recorded loss. 

## Simulated data 
The python file 'code/test_reconstruction.py' contains code used to simulate data and run each algorithm as described in the paper. The jupyter notebook 'test_reconstruction.ipynb' contains code to graph and analyze results.

## Algorithm evaluation 
Utilities for algorithm evaluation on simulated data are stored in the evaluation folder, most importantly under the reconstruction_evaluation.py file.

## Other notes
The SCoNE_parallel() function has n_jobs=1 as default, meaning the initializations run sequentially rather than in parallel. This is often preferable for large problems because it allows NumPy’s internal multithreading to fully utilize available CPU cores for each run. For smaller problems, increasing n_jobs can improve performance by running multiple initializations in parallel.

## Common pitfalls
Here are a few additional setup instructions to keep in mind:
1. Make sure that the features of Z are generally on the same scale. For example, if Z consists of admixture fractions, sex at birth (binary), and year of birth, make sure to min-max normalize year of birth so it sits on the same 0-1 scale as other variables.
2. If population structure features are admixture fractions, drop one to avoid perfect multicollienarity of features. 
3. Z should always have an intercept term.
3. For initialization, we use sklearns initialize_nmf. If the input matrices C and G are not represented as floats before input and instead are integers, sklearn random initialization will cause extreme sparsity in the initial factor matrices, which will cause poor decomposition results. The simulate_data function we use in the demo returns .astype(float).