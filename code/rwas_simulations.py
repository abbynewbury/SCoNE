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
sys.path.append('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code')
import utilities
from utilities import tensor_func
from utilities import profile_function
import sys
sys.path.append('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code/algorithms')
import CP_ALS, CP_NLS, CP_OPT, CPO_ALS1, JointMF
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
# C should consist of counts (capping count at 100 right now)
C = (C - C.min()) / (C.max() - C.min()) * 100
C = np.floor(C).astype(int) + 1
C = np.clip(C, 1, 100)

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
    all_coeffs = np.hstack([X, Z])
    all_coeffs = sm.add_constant(all_coeffs)
    model = sm.Logit(y, all_coeffs)
    result = model.fit(maxiter=100)
    gamma_true, lambda_x_true = result.params[1:X.shape[1]+1], result.params[X.shape[1]+1:]
    return gamma_true, lambda_x_true

# stack for diff labels 0/1 (weird shortcut like this since we only have two subtypes)
gamma_true1, lambda_x_true1 = fit_logistic_regression(X,Z,true_subtypes['true subtype'].to_numpy())
gamma_true0, lambda_x_true0 = fit_logistic_regression(X,Z,1-true_subtypes['true subtype'].to_numpy())
gamma_true = np.vstack([gamma_true1,gamma_true0]).T
lambda_x_true = np.vstack([lambda_x_true1,lambda_x_true0]).T
alpha_true1, lambda_c_true1 = fit_logistic_regression(C,Z,true_subtypes['true subtype'].to_numpy())
alpha_true0, lambda_c_true0 = fit_logistic_regression(C,Z,1-true_subtypes['true subtype'].to_numpy())
alpha_true = np.vstack([alpha_true1,alpha_true0]).T
lambda_c_true = np.vstack([lambda_c_true1,lambda_c_true0]).T


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

def calc_metrics(a, b, d, true_subtypes):
    # calculate clustering metrics    
    A_df = pd.DataFrame(a, columns=[f'factor{i+1}' for i in range(rank)])
    A_df['disease'] = true_subtypes['true subtype'].values
    sil_score = silhouette_score(A_df.drop('disease',axis=1).to_numpy(), A_df['disease'].values)
    kmeans = KMeans(n_clusters=len(np.unique(true_subtypes)), n_init='auto').fit(a)
    preds = kmeans.labels_
    nmi = normalized_mutual_info_score(A_df['disease'].values, preds)
    ari = adjusted_rand_score(A_df['disease'].values, preds)
    f1 = f1_score(A_df['disease'].values, preds, average='macro')
    # calculate purity
    purity = calc_purity(A_df['disease'].values, preds)

    # calc orthogonality metrics
    ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
    ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)

    # calc abs(correlation) between B and gamma

    # calc abs(correlation) between D and alpha

    return sil_score, nmi, ari, f1, purity

#3. RUN MODELS ON DATA 
# run five trials with different random initializations
for random_init in range(1): # TODO - change back to 5

    # ensure that they start at same random init
    a_init = np.random.random((T.shape[0], rank))
    b_init = np.random.random((T.shape[1], rank))
    d_init = np.random.random((T.shape[2], rank))

    # only used for joint NMF for now - can be incorporated into tensor
    lambda_x_init = np.random.random((S, Z.shape[1]))
    lambda_c_init = np.random.random((Q, Z.shape[1]))

    # 2. RUN MODELS ON DATA
    # ALS
    model_name = f'ALS_{sim_id}_{random_init}'
    ## Open tensor board writer
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    # a, b, d, rel_error, success,message,max_mem, cpu_time, user_time = profile_function(CP_ALS.cp_als,T, T_norm, rank, a_init, b_init, d_init, max_iter=50, tol=1e-4) OLD
    a, b, d, success,message, max_mem, cpu_time, user_time = profile_function(CP_ALS.cp_als,T, T_norm, rank, a_init, b_init, d_init, writer, max_iter=50, tol=1e-4) 
    sil_score, nmi, ari, f1, purity, ortho_norm_b, ortho_norm_d = calc_metrics(a,b,d,true_subtypes)
    hyperparams = {"MODEL":"ALS","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                   "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    success = 1 if success is True else 0
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
    writer.add_hparams(hyperparams, metrics)
    writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
    np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
    np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)


    # OPT
    if newton_cg:
        # loss history has relative error, LS (least squares loss) and L_ortho (orthogonal loss) at each iteration
        for reg_param in reg_params:
            model_name = f'OPT(Newton-CG)_{sim_id}_{random_init}_{reg_param}'
            log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
            writer = SummaryWriter(log_dir)
            a, b, d, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank,  a_init, b_init, d_init, reg_param=reg_param, method='Newton-CG', writer=writer, options={'maxiter':1000, 'xtol':1e-4})
            sil_score, nmi, ari, f1, purity = calc_metrics(a,true_subtypes)
            ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
            ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)
            hyperparams = {"MODEL":"OPT(Newton-CG)","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                        "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
            if success is False:
                print(f'{model_name} has message: {message}',flush=True)
            success = 1 if success is True else 0
            metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
            writer.add_hparams(hyperparams, metrics)
            writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
            # write a,b,d as well
            np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
            np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
            np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)


    if bfgs:
        for reg_param in reg_params:
            model_name = f'OPT(BFGS)_{sim_id}_{random_init}_{reg_param}'
            log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
            writer = SummaryWriter(log_dir)
            a, b, d, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank,  a_init, b_init, d_init, reg_param=reg_param, method='BFGS', writer=writer, options={'maxiter':1000, 'gtol':1e-5})
            sil_score, nmi, ari, f1, purity = calc_metrics(a,true_subtypes)
            ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
            ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)
            hyperparams = {"MODEL":"OPT(BFGS)","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                        "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
            if success is False:
                print(f'{model_name} has message: {message}',flush=True)
            success = 1 if success is True else 0
            metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
            writer.add_hparams(hyperparams, metrics)
            writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
            # write a,b,d as well
            np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
            np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
            np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)


    for reg_param in reg_params:
        model_name = f'OPT(L-BFGS-B)_{sim_id}_{random_init}_{reg_param}'
        log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
        writer = SummaryWriter(log_dir)
        a, b, d, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank, a_init, b_init, d_init, reg_param=reg_param, method='L-BFGS-B', writer=writer, options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5})
        sil_score, nmi, ari, f1, purity = calc_metrics(a,true_subtypes)
        ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
        ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)
        hyperparams = {"MODEL":"OPT(L-BFGS-B)","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                    "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
        if success is False:
            print(f'{model_name} has message: {message}',flush=True)
        success = 1 if success is True else 0
        metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
        writer.add_hparams(hyperparams, metrics)
        writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
        # write a,b,d as well
        np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
        np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
        np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)


    # CPO-ALS1
    model_name = f'CPO-ALS1_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    a, b, d, rel_error, success, message, max_mem, cpu_time, user_time = profile_function(CPO_ALS1.cpo_als1, T, T_norm, rank,  a_init, b_init, d_init, writer, max_iter=50, tol=1e-4)
    sil_score, nmi, ari, f1, purity = calc_metrics(a,true_subtypes)
    ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
    ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)
    hyperparams = {"MODEL":"CPO-ALS1","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    success = 1 if success is True else 0
    if success is False:
        print(f'{model_name} has message: {message}',flush=True)
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
    writer.add_hparams(hyperparams, metrics)
    print(a.shape,flush=True)
    writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
    np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
    np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)

    # JointMF
    model_name = f'JointMF_{sim_id}_{random_init}'
    log_dir = os.path.join(f'{simulation_results_path}/logs/tensorboard', model_name)
    writer = SummaryWriter(log_dir)
    a,b,d,lambda_x, lambda_c, loss_history, success, message, max_mem, cpu_time, user_time = profile_function(JointMF.jmf, X, C, Z, rank, a_init, b_init, d_init, lambda_x_init, lambda_c_init, method='L-BFGS-B', writer=writer, options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5})
    sil_score, nmi, ari, f1, purity = calc_metrics(a,true_subtypes)
    ortho_norm_b = np.linalg.norm(b.T@b - np.identity(rank))/np.linalg.norm(b.T@b)
    ortho_norm_d = np.linalg.norm(d.T@d - np.identity(rank))/np.linalg.norm(d.T@d)
    hyperparams = {"MODEL":"JointMF","SIM":sim_id,"INIT":random_init,"N":N, "S":S, "Q":Q, "p[0]":p[0], "S_null":s[0], "S_hom":s[1], "S_het":s[2], "pge":pge, "snp_hom_effects":snp_hom_effects,
                "snps_af_range_min":snps_af_range[0], "snps_af_range_max":snps_af_range[1], "mus_variance":mus_variance}
    success = 1 if success is True else 0
    if success is False:
        print(f'{model_name} has message: {message}',flush=True)
    metrics = {'Silhouette score': sil_score, 'NMI': nmi, 'ARI': ari, 'F1': f1, 'Purity': purity, 'B rel. orth. norm':ortho_norm_b,'D rel. orth. norm.': ortho_norm_d, "Max mem (MB)": max_mem, "CPU time":cpu_time, "User time":user_time, "Success": success}
    writer.add_hparams(hyperparams, metrics)
    print(a.shape,flush=True)
    writer.add_embedding(torch.tensor(a), metadata=metadata, metadata_header=["Subtype","Subpop"]) # add embedding of A
    # write a,b,d as well
    np.save(f'{simulation_results_path}/factor_matrices/a_{model_name}.npy', a) 
    np.save(f'{simulation_results_path}/factor_matrices/b_{model_name}.npy', b)
    np.save(f'{simulation_results_path}/factor_matrices/d_{model_name}.npy', d)

sys.stdout.flush()
sys.stderr.flush()
stdout_file.close()
stderr_file.close()