
import numpy as np

# CODE TO SIMULATE DATA
def proj_nonneg(x):
    # If already feasible, return x as-is (avoids an allocation most iterations)
    if x.min() >= 0:
        return x
    return np.maximum(x, 0)

def make_correlated_matrices(m, r, rho, seed=1):
    rng = np.random.default_rng(seed)
    Z = proj_nonneg(rng.normal(size=(m, r)))
    E = proj_nonneg(rng.normal(size=(m, r)))
    W1 = Z
    W2 = rho * Z + np.sqrt(1 - rho**2) * E
    if rho==1:
        assert np.allclose(W1, W2)
    return W1, W2

def simulate_views(n=2500, num_genes=100, M_C=100, M_Z=3, rank=3, seed=0, ZU_weight=1, noise=0, sparsity=0, rho=1, Z_noise=0):
    """
    G:  n x num_genes  (Poisson with rate WH_g.T + ZU_g.T)
    C:  n x M_C  (Poisson with rate WH_c.T + ZU_c.T)

    Z: n x M_Z covariates (non-negative random numbers)

      W   : n x rank
      H_g : num_genes x rank
      U_g : M_Z x num_genes
      H_c: n x M_C
      U_c: M_Z x M_C
    """
    rng = np.random.default_rng(seed)

    # 1. Sample factor matrices
    # Sample nonnegative factors so Poisson rates are valid
    W_C, W_G = make_correlated_matrices(n, rank, rho,seed=seed+1)
    H_C = proj_nonneg(rng.normal(size=(M_C, rank)))
    # generate linked markers
    H_G = proj_nonneg(rng.normal(size=(num_genes, rank)))
    # generate superpop shift for each feature
    U_C = proj_nonneg(rng.normal(size=(M_C, M_Z)))
    U_G =  proj_nonneg(rng.normal(size=(num_genes, M_Z)))

    # generate M_Z superpopulations
    superpops = rng.permutation(n) % M_Z   
    true_Z = (superpops[:, None] == np.arange(M_Z)).astype(int) 
    # add noise to individuals ALL superpopulations - TODO: check (Z is not a perfect approximation)
    Z_noise = Z_noise * np.random.rand(*true_Z.shape)
    Z = true_Z.astype(float).copy()
    Z += Z_noise * np.random.rand(*true_Z.shape)
    Z /= Z.sum(axis=1, keepdims=True)
    assert np.allclose(Z.sum(axis=1), 1, atol=1e-8) # all samples in exactly one group

    #2. impose sparsity (P(W_ij=0)=sparsity)
    mask = np.random.rand(*(M_C,rank)) > sparsity
    H_C = H_C * mask
    mask = np.random.rand(*(num_genes,rank)) > sparsity
    H_G = H_G * mask


    avg_corr = np.mean([np.corrcoef(W_C[:,k], W_G[:,k])[0,1]
                    for k in range(W_C.shape[1])])
    assert np.abs(avg_corr - rho) < 0.1
    
    # Means
    M_c = W_C@H_C.T + (ZU_weight)*true_Z@U_C.T + (noise)*proj_nonneg(rng.normal(size=(n, M_C)))
    M_g = W_G@H_G.T + (ZU_weight)*true_Z@U_G.T + (noise)*proj_nonneg(rng.normal(size=(n, num_genes)))

    # Generate the two observed matrices
    G = rng.poisson(M_g).astype(float)  
    C = rng.poisson(M_c).astype(float)     

    return {"G": G, "C": C, "Z":Z, "W_C": W_C, "W_G":W_G, "H_G": H_G, "H_C": H_C, "U_G": U_G, "U_C": U_C, "true_Z":true_Z}

