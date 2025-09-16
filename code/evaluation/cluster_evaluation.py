from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, f1_score, precision_score, recall_score, normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

# For subtype membership
def compute_sim_metrics(factor_matrices,ground_truth):
    # factor matrices & ground truth should be dict with "W" as key
    W_true = ground_truth["W"]
    W = factor_matrices["W"]
    # 1) Build similarity matrix S 4x2 to designate column of W to subgroup
    S = np.empty((2, 2), dtype=float)

    # i is true subgroup designation, j is
    for i in range(2): # only let it be one of the genetically informed subgroups
        for j in range(2):
            cos_sim = (np.dot(W_true[:, i],W[:, j]))/(np.linalg.norm(W_true[:, i])*np.linalg.norm(W[:, j]))
            S[i, j] = cos_sim
    # assign match up between row and col
    row_ind, col_ind = linear_sum_assignment(-S)
    #assert set(np.unique(row_ind)) == {0, 1}, "best matching subgroup is not a genetically informed subgroup"


    # return results df
    results = []
    eps = 1e-12
    P = W / (W.sum(axis=1, keepdims=True) + eps)
    results = {}
    for i_true, j_pred in zip(row_ind, col_ind):
        y = W_true[:, i_true]
        p = P[:, j_pred]

        # Probabilistic metrics (robust to class imbalance)
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p) if y.min() != y.max() else np.nan
        brier = brier_score_loss(y, p)

        # chose best threshold by f1 score
        precs, recs, thresh = precision_recall_curve(y, p)
        valid = (thresh > 0) & (thresh < 1)
        f1_scores = 2 * (precs[:-1][valid] * recs[:-1][valid]) / (precs[:-1][valid] + recs[:-1][valid])
        best_idx = np.nanargmax(f1_scores)
        best_threshold = thresh[valid][best_idx]
        print(best_threshold)

        
        yhat = (p >= best_threshold).astype(int)
        f1  = f1_score(y, yhat) if y.min() != y.max() else np.nan
        prec = precision_score(y, yhat, zero_division=0)
        rec  = recall_score(y, yhat, zero_division=0)
        nmi = normalized_mutual_info_score(y,yhat)

        
        results[f"cosine_sim_{i_true}"] = S[i_true, j_pred]
        results[f"AUROC_{i_true}"] = auroc
        results[f"AUPRC_{i_true}"] = auprc
        results[f"Brier_{i_true}"] = brier
        results[f"F1_{i_true}"] = f1
        results[f"Precision_{i_true}"] = prec
        results[f"Recall_{i_true}"] = rec
        results[f"NMI_{i_true}"] = nmi

    return results