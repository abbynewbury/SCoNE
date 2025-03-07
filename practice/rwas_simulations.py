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
import CP_ALS, CP_NLS, CP_OPT, CPO_ALS1
sys.path.append('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code/simulations')
import RGWAS_sim
import numpy as np
import pandas as pd
from plotnine import *
import pyttb as pyttb
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.metrics import silhouette_score
import json
import argparse
np.random.seed(1234)

# read in params
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
parser.add_argument("--reg_params", type=str, default='0', help="regularization parameter for OPT")
args = parser.parse_args()

# Redirect standard output and standard error
stdout_file = open(f"/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/practice/output.txt", "w")
sys.stdout = stdout_file
stderr_file = open(f"/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/practice/errors.txt", "w")
sys.stderr = stderr_file

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
reg_params = [float(i) for i in args.reg_params.split(' ')]

# read in params


# 1. SIMULATE DATA
C,X,snp_metadata, trait_metadata, true_subtypes,alpha = RGWAS_sim.generate_sim_data(N,S,Q,rank,p,s,pge,snp_hom_effects,snps_af_range,mus_variance)
X = (X - X.mean())/(X.std())
C = (C - C.mean())/(C.std())
T = np.fromfunction(lambda i, j, k: tensor_func(i, j, k, X, C), (N, S,Q), dtype=int) # assume no strat for now, exclud M_Z
T_norm = np.linalg.norm(T)


# 2. SETUP DATA RECORDING AND INITIALIZATION
metadata_df = [] # data for analysis and graphs
metadata_df_columns = ['sim_id', 'init', 'model_id', 'N','S','Q','p','s','pge','snp_hom_effects','snps_af_range','mus_variance', 'relative error', 'silhouette score', 'orthogonality norm',
                   'max memory (MB)', 'CPU time', 'user time', 'success', 'reg_param']
rel_error_history_dict = {} # structured like sim_id: model_id: rel_error
loss_history_dict = {} # specific to LS and L_ortho loss for OPT (and possibly NLS) sim_id: model_id: {LS:, L_ortho:}

# function for silhouette score metric
# calculate metrics
def calc_sil_score(a, true_subtypes):
    A_df = pd.DataFrame(a, columns=[f'factor{i+1}' for i in range(rank)])
    A_df['disease'] = true_subtypes['true subtype'].values
    sil_score = silhouette_score(A_df.drop('disease',axis=1).to_numpy(), A_df['disease'].values)
    return sil_score

rel_error_history_dict[sim_id] ={}
loss_history_dict[sim_id] ={}
# run five trials with different random initializations
for random_init in range(5):
    loss_history_dict[sim_id][f'OPT (BFGS){random_init}'] = {}
    loss_history_dict[sim_id][f'OPT (L-BFGS-B){random_init}'] = {}

    # ensure that they start at same random init
    a_init = np.random.random((T.shape[0], rank))
    b_init = np.random.random((T.shape[1], rank))
    d_init = np.random.random((T.shape[2], rank))

    # 2. RUN MODELS ON DATA
    # ALS
    a, b, d, rel_error, max_mem, cpu_time, user_time = profile_function(CP_ALS.cp_als,T, T_norm, rank, a_init, b_init, d_init, max_iter=50, tol=1e-4)
    sil_score = calc_sil_score(a,true_subtypes)
    ortho_norm= np.linalg.norm(np.multiply(b.T@b, d.T@d) - np.identity(rank))
    # record data
    metadata_df.append([sim_id,random_init,f'ALS',N,S,Q,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,rel_error[-1], sil_score, ortho_norm, max_mem, cpu_time, user_time, True, None])
    rel_error_history_dict[sim_id][f'ALS{random_init}'] = rel_error
    print('Done with ALS', flush=True)

    # OPT
    if newton_cg:
        loss_history_dict[sim_id][f'OPT (Newton-CG){random_init}'] = {}
        # loss history has relative error, LS (least squares loss) and L_ortho (orthogonal loss) at each iteration
        for reg_param in reg_params:
            a, b, d, loss_history, success, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank,  a_init, b_init, d_init, reg_param=reg_param, method='Newton-CG', options={'maxiter':1000, 'xtol':1e-4})
            sil_score = calc_sil_score(a,true_subtypes)
            ortho_norm= np.linalg.norm(np.multiply(b.T@b, d.T@d) - np.identity(rank))
            # record data
            metadata_df.append([sim_id,random_init,f'OPT (Newton-CG)',N,S,Q,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,loss_history[-1][0], sil_score, ortho_norm, max_mem, cpu_time, user_time, success, reg_param])
            rel_error_history_dict[sim_id][f'OPT (Newton-CG){random_init}'] = [i[0] for i in loss_history]
            loss_history_dict[sim_id][f'OPT (Newton-CG){random_init}']['LS'] = [i[1] for i in loss_history]
            loss_history_dict[sim_id][f'OPT (Newton-CG){random_init}']['L_ortho'] = [i[2] for i in loss_history]
            print(f'Done with OPT (Newton-CG), reg_param:{reg_param}', flush=True)

    for reg_param in reg_params:
        a, b, d, loss_history, success, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank,  a_init, b_init, d_init, reg_param=reg_param, method='BFGS', options={'maxiter':1000, 'gtol':1e-5})
        sil_score = calc_sil_score(a,true_subtypes)
        ortho_norm= np.linalg.norm(np.multiply(b.T@b, d.T@d) - np.identity(rank))
        # record data
        metadata_df.append([sim_id,random_init,f'OPT (BFGS)',N,S,Q,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,loss_history[-1][0], sil_score, ortho_norm, max_mem, cpu_time, user_time, success, reg_param])
        rel_error_history_dict[sim_id][f'OPT (BFGS){random_init}'] = [i[0] for i in loss_history]
        loss_history_dict[sim_id][f'OPT (BFGS){random_init}']['LS'] = [i[1] for i in loss_history]
        loss_history_dict[sim_id][f'OPT (BFGS){random_init}']['L_ortho'] = [i[2] for i in loss_history]
        print(f'Done with OPT (BFGS), reg_param:{reg_param}', flush=True)

    for reg_param in reg_params:
        a, b, d, loss_history, success, max_mem, cpu_time, user_time = profile_function(CP_OPT.cp_opt, T, T_norm, rank, a_init, b_init, d_init, reg_param=reg_param, method='L-BFGS-B', options={'maxcor':5, 'maxiter':1000, 'gtol':1e-5, 'maxls':10, 'ftol':1e-5})
        sil_score = calc_sil_score(a,true_subtypes)
        ortho_norm= np.linalg.norm(np.multiply(b.T@b, d.T@d) - np.identity(rank))
        # record data
        metadata_df.append([sim_id,random_init,f'OPT (L-BFGS-B)',N,S,Q,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,loss_history[-1][0], sil_score, ortho_norm, max_mem, cpu_time, user_time, success, reg_param])
        rel_error_history_dict[sim_id][f'OPT (L-BFGS-B){random_init}'] = [i[0] for i in loss_history]
        loss_history_dict[sim_id][f'OPT (L-BFGS-B){random_init}']['LS'] = [i[1] for i in loss_history]
        loss_history_dict[sim_id][f'OPT (L-BFGS-B){random_init}']['L_ortho'] = [i[2] for i in loss_history]
        print(f'Done with OPT (L-BFGS-B), reg_param:{reg_param}', flush=True)


    # CPO-ALS1
    a, b, d, rel_error, max_mem, cpu_time, user_time = profile_function(CPO_ALS1.cpo_als1, T, T_norm, rank,  a_init, b_init, d_init, max_iter=50, tol=1e-4)
    sil_score = calc_sil_score(a,true_subtypes)
    ortho_norm= np.linalg.norm(np.multiply(b.T@b, d.T@d) - np.identity(rank))
    # record data
    metadata_df.append([sim_id,random_init,f'CPO-ALS1',N,S,Q,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,rel_error[-1], sil_score, ortho_norm, max_mem, cpu_time, user_time, True, None])
    rel_error_history_dict[sim_id][f'CPO-ALS1{random_init}'] = rel_error
    print('Done with OPT (CPO-ALS1)', flush=True)

print('writing data', flush=True)
metadata_df = pd.DataFrame(metadata_df, columns = metadata_df_columns)
metadata_df.to_csv(f'{simulation_results_path}/{sim_id}_metadata.csv')
with open(f"{simulation_results_path}/{sim_id}_rel_error_history_dict.json", "w") as json_file:
    json.dump(rel_error_history_dict, json_file)
with open(f"{simulation_results_path}/{sim_id}_loss_history_dict.json", "w") as json_file:
    json.dump(loss_history_dict, json_file)

sys.stdout.flush()
sys.stderr.flush()
stdout_file.close()
stderr_file.close()