import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cophenet

# CODE TO EVALUATE RECONSTRUCTION
def frobenius_cosine_similarity(A, B):
    num = np.trace(A.T @ B)
    den = np.sqrt(np.trace(A.T @ A)) * np.sqrt(np.trace(B.T @ B))
    if den == 0:
        return 0.0
    return num / den

def kl_rel_error_denom(V):
    """
    Denominator used in Hsieh & Dhillon (KDD 2011):
    I-divergence between V and row-mean baseline Vbar,
    where each row i has constant value equal to its row mean.
    """
    V_safe = V + 1e-12
    row_means = V.mean(axis=1, keepdims=True) + 1e-12
    return np.sum(V_safe * np.log(V_safe / row_means))

def best_permutation_similarity(W_true, W_hat):
    # Compute pairwise column inner products (the numerator terms)
    M = W_true.T @ W_hat 
    row_ind, col_ind = linear_sum_assignment(-M) # maximize numerator
    W_hat_perm = W_hat[:, col_ind] # optimal permutation
    return frobenius_cosine_similarity(W_true, W_hat_perm), col_ind

def calculate_ccc(W_list):
    '''as defined by Brunet et al. 
    W_list should be sample clusters from random intializations (i.e. len(W_list) = num_init)
    '''
    consensus = None 
    for W in W_list:
        # assign hard clusters
        labels = np.argmax(W, axis=1)
        if consensus is None:
            consensus = np.zeros((len(labels), len(labels)), dtype=float)
        # Co-membership matrix for this run
        consensus += (labels[:, None] == labels[None, :]).astype(float)
    assert len(W_list) > 0
    consensus /= len(W_list) # get average over runs
    dist_condensed = squareform(1.0 - consensus, checks=False)
    HC_linkage = linkage(dist_condensed, method="average")
    coph_corr, _ = cophenet(HC_linkage, dist_condensed)
    return coph_corr