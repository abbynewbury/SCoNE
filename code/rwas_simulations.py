#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=rwas_sim
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


import sys
code_path = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code' # TODO - don't hardcode
sys.path.append(code_path)
import utilities
from utilities import tensor_func
from utilities import profile_function
import sys
sys.path.append('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code/algorithms')
import CP_ALS, CP_NLS, CP_OPT, CPO_ALS1, JointMF, CP_OPT_adj
sys.path.append('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code/simulations')
import RGWAS_sim
import numpy as np
import pandas as pd
#from plotnine import *
from sklearn.metrics import silhouette_score
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# Fix tensorboard problem
# import tensorflow as tf
# import tensorboard as tb
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from collections import Counter
from sklearn.decomposition import PCA
import statsmodels.api as sm
import subprocess



np.random.seed(1234)

### PARAMETERS

parser = argparse.ArgumentParser(description="Read arguments from shell script")
parser.add_argument("--N", type=int, required=True, help="Input argument value")
parser.add_argument("--S", type=int, required=True, help="Input argument value")
parser.add_argument("--Q", type=int, required=True, help="Input argument value")
parser.add_argument("--p", type=str, required=True, help="Input argument value")
parser.add_argument("--s", type=str, required=True, help="Input argument value")
parser.add_argument("--pge", type=float, required=True, help="Input argument value")
parser.add_argument("--snp_hom_effects", type=str, required=True, help="Input argument value")
parser.add_argument("--snps_af_range", type=str, required=True, help="Input argument value")
parser.add_argument("--mus_variance", type=float, required=True, help="Input argument value")
parser.add_argument("--sim_id", type=int, required=True, help="ID representing simulation params")
parser.add_argument("--simulation_results_path", type=str, required=True, help="path to write results")
parser.add_argument("--newton_cg", type=bool, default=False, help="whether to use Newton-CG as OPT solver, very slow")
parser.add_argument("--bfgs", type=bool, default=False, help="whether to use BFGS as OPT solver, often fails to succeed due to hitting max # iterations")
parser.add_argument("--reg_params", type=str, default='0', help="regularization parameter for OPT")
parser.add_argument("--num_pops", type=int, default=1, help="number of subpopulations")
parser.add_argument("--r_path", type=str, default='', help="path to r") #/gpfs/commons/home/anewbury/miniconda/bin/Rscript
args = parser.parse_args()

rank=2 # same as number of subtypes
N = args.N
S = args.S
Q = args.Q
p = [float(i) for i in args.p.split(' ')]
s = [int(i) for i in args.s.split(' ')]
pge = args.pge
snp_hom_effects = args.snp_hom_effects
snps_af_range = [float(i) for i in args.snps_af_range.split(' ')]
mus_variance = args.mus_variance
sim_id = args.sim_id
simulation_results_path = args.simulation_results_path
newton_cg = args.newton_cg
bfgs = args.bfgs
num_pops = args.num_pops
reg_params = [float(i) for i in args.reg_params.split(' ')]
r_path = args.r_path

# Redirect standard output and standard error
stdout_file = open(f"{simulation_results_path}/logs/outputs/{sim_id}_output.txt", "w")
sys.stdout = stdout_file
stderr_file = open(f"{simulation_results_path}/logs/errors/{sim_id}_error.txt", "w")
sys.stderr = stderr_file

### PARAMETERS

# 1. SIMULATE DATA
# Use: C,X,snp_metadata, trait_metadata, true_subtypes,true_subpops = generate_sim_data(N,S,Q,K,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,num_pops)
C,X,snp_metadata, trait_metadata, true_subtypes, true_subpops = RGWAS_sim.generate_sim_data(N,S,Q,rank,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,num_pops)
# X should consist of 0s or 1s (dominant model right now)
X = np.where((X == 1) | (X == 2), 1, 0) # dominant model

# keep condition counts up to 10
C = (C - C.min()) / (C.max() - C.min()) * 10
C = np.floor(C).astype(int)
C = np.clip(C, 0, 10)

# set Z to represent top 5 PCs of genotype matrix - proxy for subpopulation
X_scaled = (X - X.mean())/(X.std())
X_pca = PCA(n_components=5)
Z = X_pca.fit_transform(X_scaled)

# generate tensor
T = np.fromfunction(lambda i, j, k: tensor_func(i, j, k, X, (C - C.min())/(C.max() - C.min())), (N, S,Q), dtype=int)
T_norm = np.linalg.norm(T)
metadata = true_subtypes.join(true_subpops)
metadata = metadata[['true subtype','true subpop']].values.tolist()

# 1.5. GET GROUND TRUTH
def fit_logistic_regression(X, Z, y):
    gamma_true = []
    gamma_true_pvals = []
    for i in range(X.shape[1]):
        all_coeffs = np.hstack([X[:,i].reshape(-1, 1), Z])
        all_coeffs = sm.add_constant(all_coeffs)
        model = sm.Logit(y, all_coeffs)
        result = model.fit(maxiter=500)
        assert result.mle_retvals['converged'], "Model failed to converge"
        gamma_true.append(result.params[1]) # 1 bc 0 is coeff
        gamma_true_pvals.append(result.pvalues[1])
    return np.array(gamma_true), np.array(gamma_true_pvals)

# stack for diff labels 0/1 (weird shortcut like this since we only have two subtypes)
gamma, gamma_pvals = fit_logistic_regression(X,Z,true_subtypes['true subtype'].to_numpy())
gamma = np.vstack([gamma,-gamma]).T
gamma_pvals = np.vstack([gamma_pvals,gamma_pvals]).T

alpha, alpha_pvals = fit_logistic_regression(C,Z,true_subtypes['true subtype'].to_numpy())
alpha = np.vstack([alpha,-alpha]).T
alpha_pvals = np.vstack([alpha_pvals,alpha_pvals]).T


# 2. FUNCTIONS TO CALCULATE METRICS
# calculate metrics
def calc_purity(labels, preds):
    cluster_labels = {}
    for pred, true in zip(preds, labels):
        # like cluster_label = {0:[true vals of people in cluster 0], ...}
        cluster_labels.setdefault(pred, []).append(true)
    correct = 0 
    for cluster in cluster_labels.values():
        most_common = Counter(cluster).most_common(1)[0] # tuble of label that is most common and count
        correct += most_common[1]
    purity = correct/len(labels)
    return purity

def calc_cluster_metrics(a, true_vals):
    # calculate cluster metrics for sample loadings (a)
    if len(set(true_vals))>1:
        sil_score = silhouette_score(a, true_vals)
        kmeans = KMeans(n_clusters=len(np.unique(true_vals)), n_init='auto').fit(a)
        preds = kmeans.labels_
        nmi = normalized_mutual_info_score(true_vals, preds)
        ari = adjusted_rand_score(true_vals, preds)
        # calculate purity
        purity = max(calc_purity(true_vals, preds),calc_purity(true_vals, 1-preds))
    else:
        sil_score, nmi, ari, purity = [float('nan')]*4
    return sil_score, nmi, ari, purity


def calc_metrics(a, b, d,true_vals, gamma, alpha):
    # gamma - oracle SNP effect sizes
    # alpha - oracle condition effect sizes
    # calculate clustering metrics    
    sil_score, nmi, ari, purity = calc_cluster_metrics(a, true_vals)

    # calc orthogonality metrics
    ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
    ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)

    # calc abs(correlation) between B and gamma - since only two subtypes and taking abs correlation - doesn't matter which column of B is paired up with which classification of subtypes (subtypes or 1-subtypes)
    if set(np.unique(b)).issubset({0, 1}):
        b_scaled = b
    else:
        b_scaled = (b - b.mean(axis=0)) / b.std(axis=0)
    gamma_scaled = (gamma - gamma.mean(axis=0)) / gamma.std(axis=0)
    pearson_r_b_gamma = np.dot(b_scaled.T, gamma_scaled) / b_scaled.shape[0]

    # calc abs(correlation) between D and alpha - since only two subtypes and taking abs correlation - doesn't matter which column of B is paired up with which classification of subtypes (subtypes or 1-subtypes)
    if set(np.unique(b)).issubset({0, 1}):
        d_scaled = d
    else:
        d_scaled = (d - d.mean(axis=0)) / d.std(axis=0)
    alpha_scaled = (alpha - alpha.mean(axis=0)) / alpha.std(axis=0)
    pearson_r_d_alpha = np.dot(d_scaled.T, alpha_scaled) / d_scaled.shape[0]
    return sil_score, nmi, ari, purity,ortho_norm_b,ortho_norm_d, abs(pearson_r_b_gamma[:,0]).mean(), abs(pearson_r_d_alpha[:,0]).mean()

#3. RUN MODELS ON DATA 
# run five trials with different random initializations
for random_init in range(1): # TODO: switch back to 5

    # ensure that they start at same random init
    A_init = np.random.random((T.shape[0], rank))
    B_init = np.random.random((T.shape[1], rank))
    D_init = np.random.random((T.shape[2], rank))

    # used to initialize adjustment matrices (in JointMF, ALS, OPT)
    B_prime_init = np.random.random((S, Z.shape[1]))
    D_prime_init = np.random.random((Q, Z.shape[1]))

    # 2. RUN MODELS ON DATA
    # ALS
    model_name = f'ALS_{sim_id}_{random_init}'
    ## Open tensor board writer
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    A, B, D, success,message, max_mem, cpu_time, user_time = profile_function(CP_ALS.cp_als,T, T_norm, rank, A_init, B_init, D_init, writer, max_iter=50, tol=1e-4,mem_target='function') 
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"ALS","SIM":sim_id,"INIT":random_init,"Adjusted":0,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                   "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # CLIGEN (no cov. adjustment, C in {0,1})
    T_cligen = np.fromfunction(lambda i, j, k: tensor_func(i, j, k, X, (C > 0).astype(int)), (N, S,Q), dtype=int)
    T_cligen_norm = np.linalg.norm(T_cligen)
    model_name = f'CLIGEN_{sim_id}_{random_init}'
    ## Open tensor board writer
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    A, B, D, success,message, max_mem, cpu_time, user_time = profile_function(CP_ALS.cp_als,T_cligen, T_cligen_norm, rank, A_init, B_init, D_init, writer, max_iter=50, tol=1e-4,mem_target='function') 
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"CLIGEN","SIM":sim_id,"INIT":random_init,"Adjusted":0,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                   "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)


    # OPT
    for reg_param in reg_params:
        if newton_cg:
            solver = 'Newton-CG' 
            model_name = f'OPT({solver})_{sim_id}_{random_init}_{reg_param}'
            method=solver
            options={'maxiter':1000, 'xtol':1e-4}

        if bfgs:
            solver = 'BFGS'
            model_name = f'OPT({solver})_{sim_id}_{random_init}_{reg_param}'
            method=solver
            options={'maxiter':1000, 'gtol':1e-5}

        else: # default L-BFGS
            solver = 'L-BFGS-B'
            model_name = f'OPT({solver})_{sim_id}_{random_init}_{reg_param}'
            method=solver
            options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5}


        log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
        writer = SummaryWriter(log_dir)
        A, B, D, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank, A_init, B_init, D_init, reg_param=reg_param, method=method, writer=writer, options=options,mem_target='function')
        sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
        sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
        hyperparams = {"MODEL":f"OPT({solver})","SIM":sim_id,"INIT":random_init,"Adjusted":0,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                    "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
        metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
                'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                    "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
        writer.add_hparams(hyperparams, metrics)
        writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
        # write a,b,d as well
        np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
        np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
        np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # OPT adjusted (use same solver as above)
    for reg_param in reg_params:
        model_name = f'OPT_adj({solver})_{sim_id}_{random_init}_{reg_param}'
        log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
        writer = SummaryWriter(log_dir)
        A, B, D, B_prime, D_prime, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(CP_OPT_adj.cp_opt_adjusted, T, Z, T_norm, rank, A_init, B_init, D_init,B_prime_init, D_prime_init, reg_param=reg_param, method=method, writer=writer, options=options,mem_target='function')
        sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
        sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
        hyperparams = {"MODEL":f"OPT_adj({solver})","SIM":sim_id,"INIT":random_init,"Adjusted":1,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                    "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
        metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
                'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                    "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
        writer.add_hparams(hyperparams, metrics)
        writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
        # write a,b,d as well
        np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
        np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
        np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)


    # CPO-ALS1
    model_name = f'CPO-ALS1_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    A, B, D, rel_error, success, message, max_mem, cpu_time, user_time = profile_function(CPO_ALS1.cpo_als1, T, T_norm, rank,  A_init, B_init, D_init, writer, max_iter=50, tol=1e-4,mem_target='function')
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"CPO-ALS1","SIM":sim_id,"INIT":random_init,"Adjusted":0,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # JointMF
    model_name = f'JointMF_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    A,B,D,B_prime, D_prime, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(JointMF.jmf, X, C, Z, rank, A_init, B_init, D_init, B_prime_init, D_prime_init, method='L-BFGS-B', writer=writer, options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5},mem_target='function')
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"JointMF","SIM":sim_id,"INIT":random_init,"Adjusted":1,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # HNMF (but different losses)
    model_name = f'HNMF_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    A,B,D,B_prime, D_prime, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(JointMF.jmf, X, C, np.zeros((Z.shape[0],Z.shape[1])), rank, A_init, B_init, D_init, B_prime_init, D_prime_init, method='L-BFGS-B', writer=writer, options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5},mem_target='function')
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"HNMF","SIM":sim_id,"INIT":random_init,"Adjusted":1,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # Multi-view biclustering - will only run once since only two subtypes (for now)
    model_name = f'MVBC_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    # running model
    np.save(f'{simulation_results_path}/X_{sim_id}_{random_init}.npy',X.astype(np.float64))
    np.save(f'{simulation_results_path}/C_{sim_id}_{random_init}.npy',C.astype(np.float64))
    result, max_mem, cpu_time, user_time = profile_function(lambda: subprocess.run(
    f'{r_path} {code_path}/algorithms/MVBC.R {simulation_results_path}/X_{sim_id}_{random_init}.npy {simulation_results_path}/C_{sim_id}_{random_init}.npy',
    stdout=subprocess.PIPE,
    text=True,
    shell=True, executable='/bin/bash'),mem_target='subprocess')
    # Parse the JSON output (result["Cluster"])
    r_output = result.stdout.strip()
    parsed = json.loads(r_output)
    success=True
    A = np.array(parsed['Cluster']).flatten().reshape(-1, 1)
    B = np.array(parsed['FeatClusters'][0]).flatten().reshape(-1, 1)
    D = np.array(parsed['FeatClusters'][1]).flatten().reshape(-1, 1)
    os.remove(f'{simulation_results_path}/X_{sim_id}_{random_init}.npy')
    os.remove(f'{simulation_results_path}/C_{sim_id}_{random_init}.npy')
    # running model
    sil_score, nmi, ari, purity, ortho_norm_b, ortho_norm_d, pearson_r_b_gamma, pearson_r_d_alpha = calc_metrics(A,B,D,true_subtypes['true subtype'].values, gamma, alpha) # clustering per true subtypes
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(A, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"MVBC","SIM":sim_id,"INIT":random_init,"Adjusted":0,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':pearson_r_b_gamma, 'R2 D,Alpha':pearson_r_d_alpha, 
               'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": 1 if success is True else 0}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(A), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/A_{model_name}.npy', A) 
    np.save(f'{simulation_results_path}/factor_matrices/B_{model_name}.npy', B)
    np.save(f'{simulation_results_path}/factor_matrices/D_{model_name}.npy', D)

    # Oracle PRS
    model_name = f'OraclePRSCondition_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    sil_score, nmi, ari, purity = calc_cluster_metrics(C@alpha, true_subtypes['true subtype'].values)
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(C@alpha, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"OraclePRSCondition","SIM":sim_id,"INIT":random_init,"Adjusted":1,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':1, 'R2 D,Alpha':1, 
               'B rel. orth. norm':float('nan'),'D rel. orth. norm.': float('nan'), 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": float('nan'), "CPU time":float('nan'), "User time":float('nan'), "Success": 1}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(C@alpha), metadata=metadata, metadata_header=["Subtype","Subpop"])


    model_name = f'OraclePRSSNP_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    sil_score, nmi, ari, purity = calc_cluster_metrics(X@gamma, true_subtypes['true subtype'].values)
    sil_score_conf, nmi_conf, ari_conf, purity_conf = calc_cluster_metrics(X@gamma, true_subpops['true subpop'].values) # clustering per confounders
    hyperparams = {"MODEL":"OraclePRSSNP","SIM":sim_id,"INIT":random_init,"Adjusted":1,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'R2 B,Gamma':1, 'R2 D,Alpha':1, 
               'B rel. orth. norm':float('nan'),'D rel. orth. norm.': float('nan'), 'Silhouette score (conf)':sil_score_conf, 'NMI (conf)':nmi_conf , 'ARI (conf)':ari_conf, 'Purity (conf)':purity_conf, 
                "Max mem (MB)": float('nan'), "CPU time":float('nan'), "User time":float('nan'), "Success": 1}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(X@gamma), metadata=metadata, metadata_header=["Subtype","Subpop"])

sys.stdout.flush()
sys.stderr.flush()
stdout_file.close()
stderr_file.close()