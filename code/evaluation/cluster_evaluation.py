from sklearn.metrics import (
    normalized_mutual_info_score
)
import numpy as np
import pandas as pd
import pickle
import fcntl


# work on cluster evaluation metrics
def purity(true,pred):
    # true and pred should be binary matrices - hard clustering, true should have shape (Nxc), pred should have shape (Nxk)
    # calcualte N(i,j)
    overlapping_counts = true.T @ pred
    return np.sum(np.max(overlapping_counts,axis=0))/true.shape[0]

def entropy(labels):
    # labels should be a binary matrix with labels as columns
    prob = labels.sum(axis=0)/labels.shape[0]
    assert not np.all(prob == 0) 
    prob = prob[prob>0]
    en = np.sum((-1)*prob*np.log2(prob))
    return en

def norm_cond_entropy(true,pred):
    # true and pred should be binary matrices - hard clustering, true should have shape (Nxc), pred should have shape (Nxk)
    # represents true conditional on predicted (i.e. H(C|K) if C is true labels and K is predicted labels)
    # calculate N(i,j)
    nij = true.T @ pred
    # calculate N(j)
    nj = pred.sum(axis=0)

    frac = np.zeros_like(nij, dtype=float)
    valid = nj > 0                   # only divide where cluster has members
    frac[:, valid] = nij[:, valid] / nj[valid]

    logfrac = np.zeros_like(frac, dtype=float)
    mask = frac > 0
    logfrac[mask] = np.log2(frac[mask])

    terms = nij * logfrac
    norm_cond_en = -1/(true.shape[0]*np.log2(true.shape[1]))*np.sum(terms)
    return norm_cond_en

def normalized_mutual_info(true,pred):
    numerator = 2*(entropy(true) - (np.log2(true.shape[1])*norm_cond_entropy(true,pred))) # using own defn of normalized conditional entropt
    denominator = entropy(true) + entropy(pred)
    nmi = numerator/denominator
    # double check close to sklearn value (will also validate norm_cond_entropy)
    assert np.isclose(normalized_mutual_info_score(np.argmax(true,axis=1),np.argmax(pred,axis=1)),nmi)

    return nmi
            
def compute_sim_metrics(factor_matrices,ground_truth,confounding_matrix=None):
    # factor matrices & ground truth should be dict with "W" as key
    W_true = ground_truth["W"]
    W = factor_matrices["W"]

    # create hard clusters
    labels = np.argmax(W, axis=1)
    binary_W = np.zeros_like(W)
    binary_W[np.arange(W.shape[0]), labels] = 1

    assert np.all(W_true.sum(axis=1) == 1), "true has rows that are not one-hot"
    assert np.all(binary_W.sum(axis=1) == 1), "pred has rows that are not one-hot"

    results = {}
    # calculate purity
    results["purity"] = purity(W_true, binary_W) # 0-1, want values closer to 1

    # calculate normalized conditional entropy
    results["norm_cond_entropy"] = norm_cond_entropy(W_true,binary_W) # 0-1, want values closer to 0

    # calculate normalized mutual information
    results["nmi"] = normalized_mutual_info(W_true,binary_W) # 0-1, want values closer to 1

    if confounding_matrix is not None and confounding_matrix.shape[1]!=1:
        row_sums = np.sum(confounding_matrix, axis=1)
        # Assert that all row sums are close to 1
        assert np.allclose(row_sums, np.ones(confounding_matrix.shape[0]))
        results["nmi_w_confounder"] = normalized_mutual_info(confounding_matrix,binary_W) # 0-1, want values closer to 1

    return results