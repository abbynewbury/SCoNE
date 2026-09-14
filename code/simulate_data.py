import numpy as np

# CODE TO SIMULATE DATA
def proj_nonneg(x):
    # If already feasible, return x as-is (avoids an allocation most iterations)
    if x.min() >= 0:
        return x
    return np.maximum(x, 0)

def make_correlated_matrices(r, X, y_nonneg=False, seed=0):
    # random matrix always created from N(0,1) distribution
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=X.shape)
    if y_nonneg:
        Y = proj_nonneg(Y)
    return_mat = r*X + np.sqrt(1 - r**2) * Y
    if r==1:
        assert np.allclose(X, return_mat)
    return return_mat

def simulate_views(n=1000, M_G=20, M_C=20, 
                   rank=3, seed=0, gamma=5, noise=0,
                   sparsity=0.2, rG=0, rGC=0, rZ=0,
                   signed_cov_effects=False,subgroup_structure=True,
                   overdispersion_nu=0
                  ):
    """
    n: number of participants
    M_G: number of genetic features
    M_C: number of clinical features
    rank: number of subgroups
    seed: random seed
    gamma: ratio of marginal variance of covariate vs. subgroup component for G
    noise: weighting of random noise
    sparsity: controls sparsity of H_G and H_C
    rG: controls similarity between genetic subgroups
    rGC: controls similarity between corresponding clinical and genetic subgroups
    rZ: controls similarity between covariate and subgroup 1
    signed_cov_effects: no non-negative projection for U_G and U_C (bool)
    subgroup_structure: whether to include WHT in means of G and C (bool)
    overdispersion_nu: paramater to control overdispersion in negative binomial distribution for C, if !=0
    """

    rng = np.random.default_rng(seed)
    
    if subgroup_structure:
        # 1. Simulate W by random assignment to rank subgroups
        sizes = np.full(rank, n//rank) # evenly balance subgroups
        sizes[:n % rank] += 1
        labels = np.repeat(np.arange(rank), sizes)
        rng.shuffle(labels)
        W = np.zeros((n, rank), dtype=int)
        W[np.arange(n), labels] = 1
        
        # 2. Simulate H_G with genetic architecture similarity rG
        Sigma = (1-rG)*np.eye(rank,rank) + np.full((rank,rank),rG)
        H_G = proj_nonneg(rng.multivariate_normal(np.zeros((rank)), Sigma, size=M_G))
        
        # 3. Simulate H_C with similarity to genetic subgroups rGC
        H_C = proj_nonneg(make_correlated_matrices(rGC, H_G, seed=seed))
        
        # 4. impose sparsity (P(W_ij=0)=sparsity)
        mask = rng.random((M_C,rank)) > sparsity
        H_C = H_C * mask
        mask = rng.random((M_G,rank)) > sparsity
        H_G = H_G * mask
    
        # 5. Generate covariate with similarity to subgroup 1 structure
        z = make_correlated_matrices(rZ, W[:,1], y_nonneg=True, seed=seed)
    
    else: 
        assert rZ==0, "set rZ to 0 if no subgroup structure"
        z = proj_nonneg(rng.normal(size=n))
    
    # 5. Generate covariate matrix
    assert (z>=0).all()
    Z = np.column_stack([np.ones(len(z)), z])
    
    # 6. Simulate covariate effects
    U_C = rng.normal(size=(M_C, Z.shape[1]))
    U_G = rng.normal(size=(M_G, Z.shape[1]))
    if not signed_cov_effects:
        U_C = proj_nonneg(U_C)
        U_G = proj_nonneg(U_G)
    
    # 7. Generate matrix means
    if subgroup_structure:
        mu_C = W@H_C.T + Z@U_C.T + (noise)*proj_nonneg(rng.normal(size=(n, M_C)))
        # standardize A and B to have same marginal standard deviation
        A = W @ H_G.T
        B = Z @ U_G.T
        mu_G = A + gamma*(np.std(A) / np.std(B))*B + (noise)*proj_nonneg(rng.normal(size=(n, M_G)))
    else:
        mu_C = Z@U_C.T + (noise)*proj_nonneg(rng.normal(size=(n, M_C)))
        mu_G = Z@U_G.T + (noise)*proj_nonneg(rng.normal(size=(n, M_G)))
    
    if signed_cov_effects:
        mu_C = np.exp(mu_C)
        mu_G = np.exp(mu_G)
    
    # Generate the two observed matrices
    G = rng.poisson(mu_G).astype(float)  
    if overdispersion_nu==0:
        C = rng.poisson(mu_C).astype(float) 
    else:
        p_nb = 1 / (1 + overdispersion_nu**2 * mu_C)
        n_nb = 1 / overdispersion_nu**2
        C = rng.negative_binomial(n_nb, p_nb).astype(float) 
    if subgroup_structure:
        return_dict = {"G": G, "C": C, "Z":Z, "W": W, "H_G": H_G, "H_C": H_C, "U_G": U_G, "U_C": U_C}
    else:
        return_dict = {"G": G, "C": C, "Z":Z, "U_G": U_G, "U_C": U_C}
    return return_dict
