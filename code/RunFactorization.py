#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=RunFactorization
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=RunFactorization.txt
#SBATCH --error=RunFactorization.txt

import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
import pickle
import glob
import time
from itertools import product
from joblib import Parallel, delayed
from functools import partial
import sys
import mlflow
import submitit
import random
from mlflow.tracking import MlflowClient
from datetime import datetime
from zoneinfo import ZoneInfo


intermediate_plink_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
intermediate_saige_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_saige'
root_dir = '/gpfs/commons/datasets/1000genomes'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
sim_output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
admixture_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5'
map_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map'
code_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code'
artifact_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/output/logs'
sys.path.append(code_dir)
import simulations.genomes1000_sim as sim_functions
import algorithms.SCoNE as SCoNE
import algorithms.MLFlowWrapper as MLFlowWrapper
import evaluation.cluster_evaluation as cluster_evaluation
import importlib

# PARAMETERS
# generate all combinations of e and ps variables
ps_list = [True,False]
e_list = [0.25,0.5,0.75,1] 
g_list = [10] 
num_markers = 100
bfile_path=f'{sim_output_dir}/G'
af_df_filepath=admixture_filepath
rank = 3
tuning = True # to run sparsity tuning step
testing = True # to run testing step
# PARAMETERS
np.random.seed(42)

# read in Z
igsr_samples = sim_functions.read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{sim_output_dir}/G')
igsr_samples['Sex'] = igsr_samples['Sex'].map({'female': 0, 'male': 1})

admixture = pd.read_csv(f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q',sep='\s+',header=None)
fam_df = pd.read_csv(f'{sim_output_dir}/G.fam',sep='\s+',header=None)
fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
admixture['IID'] = fam_df['IID'].values
Z_df = admixture.merge(igsr_samples[['IID']], on='IID',how='inner') # not including covar Sex
Z = Z_df[[i for i in Z_df.columns if i!='IID']].to_numpy()



def _call_kwargs(kw):
    return run_one_wrapper(**kw)  # expands kwargs dict

def run_one_wrapper(ps, e, dataset, g, init,
                    Z,
                    W, H_G, H_C, U_G, U_C,
                    lambda_W, lambda_H_G, lambda_H_C,
                    G_loss_type, C_loss_type,
                    max_outer,min_outer,tol,nonneg,
                    sim_output_dir, exp_num,run_name,artifact_dir): 
    output_suffix = sim_functions.get_output_file_suffix(ps=ps, e=e, dataset=dataset, g=g)
    print(f"{sim_output_dir}/simulation_metadata_{output_suffix}.pkl")
    C = np.load(f'{sim_output_dir}/C_{output_suffix}.npy')
    with open(f"{sim_output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_index = fam_df[fam_df['IID'].isin(simulation_metadata['iid_order'])].index # some samples removed due to overlap btwn subgroups, need correct length
    # read in G with num_markers
    G = np.loadtxt(f'{sim_output_dir}/G_{sim_functions.get_output_file_suffix(ps,e,dataset,g)}.raw',  usecols=range(6, num_markers+6), dtype=np.int8, skiprows=1)
    fam_df_subset = pd.read_csv(f'{sim_output_dir}/G_{sim_functions.get_output_file_suffix(ps,e,dataset,g)}.fam',sep='\s+',header=None)
    fam_df_subset.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    assert all(fam_df_subset['IID'].values == fam_df['IID'].values)
    # Note: first two columns are genetically-informed subgroups by construction
    W_true = (simulation_metadata['phenotypic_subgroups'].pivot(index='IID',columns='phenotypic_subgroup',values='subgroup')
            .reindex(simulation_metadata['iid_order']).iloc[:,:2].to_numpy().astype(int))
    ground_truth = {"W":W_true}

    return MLFlowWrapper.train_with_mlflow( 
        algorithm_func=SCoNE.alternating_opt,
        artifact_dir=artifact_dir,
        run_name=run_name,
        algorithm_func_kwargs={"G":G[iid_index,:], "C":C, "Z":Z[iid_index,:], "W":W[iid_index,:], "H_G":H_G, "H_C":H_C, "U_G":U_G, "U_C":U_C,
                               "lambda_W":lambda_W, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C,
                               "method":'L-BFGS-B', "options":{'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6},  # keep scipy methods and options fixed
                               "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                               "max_outer":max_outer, "min_outer":min_outer, "tol":tol, "nonneg":nonneg},
        params={"ps": ps, "e": e, "dataset": dataset, "g": g, "init":init,
                "run_seed": simulation_metadata["run_seed"], "tol": tol, "max_outer": max_outer, "min_outer":min_outer,
                "lambda_W": lambda_W, "lambda_H_G": lambda_H_G, "lambda_H_C": lambda_H_C, "run_name":run_name,
                "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                "method":'L-BFGS-B', "options":{'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6}},
        eval_fn=cluster_evaluation.compute_sim_metrics, 
        ground_truth=ground_truth,
        experiment_name=str(exp_num))

os.makedirs(artifact_dir, exist_ok=True)
os.makedirs(f"{os.path.dirname(artifact_dir)}/slurm_logs", exist_ok=True)
os.makedirs(f"{os.path.dirname(artifact_dir)}/logs_tuning", exist_ok=True)

# PRELIMINARY: set up fixed params across experiments
exp_map = {}
tuning_dataset = {}
for i, (ps, e, g) in enumerate(product(ps_list, e_list, g_list)):
    key = (ps, e, g)
    tuning_dataset[key] = np.random.randint(0, 11)
    exp_map[key] = i
    

# STEP 1: hparam tuning with 1 randomly selected dataset per experiment (and then remove it from testing)
if tuning:
    # submit as a SLURM array (adjust params as needed)
    executor = submitit.AutoExecutor(folder=f"{os.path.dirname(artifact_dir)}/slurm_logs")
    executor.update_parameters(
        slurm_job_name="fact-grid",
        timeout_min=180,
        cpus_per_task=1,
        mem_gb=3,
        slurm_array_parallelism=200,
        stderr_to_stdout=True,
        slurm_additional_parameters={
            "output": f"{os.path.dirname(artifact_dir)}/slurm_logs/%x_%A_%a.out",
            "error":  f"{os.path.dirname(artifact_dir)}/slurm_logs/%x_%A_%a.out",
            "export": "ALL,PYTHONUNBUFFERED=1",
            "mail-type": "FAIL",                      
            "mail-user": "anewbury@nygenome.org",
        },
    )
    combos = [(ps, e, g, lW, lHG, lHC, rn)
    for ps, e, g, lW, lHG, lHC, rn in product(
        ps_list, e_list, g_list, [0,0.3,0.5,1],[0,0.3,0.5,1],[0,0.3,0.5,1], ['SCoNE','SCoNE(Fro)','sHNMF']
    ) if (lW, lHG, lHC) != (0, 0, 0)]
    W = np.random.uniform(low=0.1,high=1,size=(2504, rank))# aorund 0 to 1e-2
    H_G = np.random.uniform(low=0.1,high=1,size=(num_markers, rank))
    H_C = np.random.uniform(low=0.1,high=1,size=(100, rank)) # fixed: 100 clinical vars
    U_G = np.random.uniform(low=0.1,high=1,size=(num_markers, Z.shape[1]))
    U_C = np.random.uniform(low=0.1,high=1,size=(100, Z.shape[1]))

    cfgs = [
        dict(
            ps=ps, e=e, dataset=tuning_dataset[ps, e, g], g=g,
            init = 0,
            Z=Z if run_name != 'sHNMF' else np.zeros((Z.shape[0],Z.shape[1])), W=W, H_G=H_G, H_C=H_C, U_G=U_G, U_C=U_C,
            lambda_W=lambda_W, lambda_H_G=lambda_H_G, lambda_H_C=lambda_H_C,
            max_outer=50, min_outer=5, tol=1e-6, nonneg=True,
            sim_output_dir=sim_output_dir, exp_num=exp_map[ps, e, g],
            run_name=run_name,G_loss_type='kl_div' if run_name!='SCoNE(Fro)' else 'fro', C_loss_type='kl_div' if run_name!='SCoNE(Fro)' else 'fro',
            artifact_dir=f"{os.path.dirname(artifact_dir)}/logs_tuning"
        )
        for i, (ps, e, g, lambda_W, lambda_H_G, lambda_H_C, run_name) in enumerate(combos)
    ]
    mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/logs_tuning")
    for i in range(len(list(product(ps_list, e_list, g_list)))):
        exp = mlflow.set_experiment(str(i)) # set experiment id ahead of time for slurm parallelism
    jobs = executor.map_array(_call_kwargs, cfgs)

    # Block until every array task has completed (and raise if any failed)
    t0 = datetime.now(ZoneInfo("America/New_York"))
    _ = [j.result() for j in jobs]
    t1 = datetime.now(ZoneInfo("America/New_York"))
    print(f"[{t1:%Y-%m-%d %H:%M:%S %Z}] now running… elapsed={t1 - t0}", flush=True)

if testing:
    testing_runs = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)','sHNMF']
    if ['SCoNE','SCoNE(Fro)','sHNMF'] in testing_runs:

        # STEP 2: find optimal sparsity parameters for each method
        mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/logs_tuning")
        client = MlflowClient()
        # Get all experiments
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            df = mlflow.search_runs([exp.experiment_id])
            all_runs.append(df)
        all_runs = pd.concat(all_runs, ignore_index=True)
        all_runs['params.ps'] = all_runs["params.ps"].map({"True": True, "False": False}).astype("boolean")  # columns you want as bools (nullable)
        all_runs[['params.e','params.g','params.lambda_W','params.lambda_H_G','params.lambda_H_C']]  = all_runs[['params.e','params.g','params.lambda_W','params.lambda_H_G','params.lambda_H_C']].astype("float64") 
        all_runs['G_plus_C_loss'] = all_runs['metrics.G_loss'] + all_runs['metrics.C_loss']
        idx = all_runs.groupby(["params.run_name", "params.ps", "params.e", "params.g"])["G_plus_C_loss"].idxmin()
        best = (
            all_runs.loc[idx, all_runs.columns]
            .sort_values(["params.run_name", "params.ps", "params.e", "params.g"])
            .reset_index(drop=True)
        )

    # STEP 3: testing/comparison over 10 other datasets for each experiment
    # submit as a SLURM array (adjust params as needed)
    executor = submitit.AutoExecutor(folder=f"{os.path.dirname(artifact_dir)}/slurm_logs")
    executor.update_parameters(
        slurm_job_name="fact-grid",
        timeout_min=180,
        cpus_per_task=1,
        mem_gb=3,
        slurm_array_parallelism=200,
        stderr_to_stdout=True,
        slurm_additional_parameters={
            "output": "slurm_logs/%x_%A_%a.out",
            "error":  "slurm_logs/%x_%A_%a.out",
            "export": "ALL,PYTHONUNBUFFERED=1",
        },
    )
    # build combos excluding the tuning dataset per (ps,e,g)
    combos = []
    for ps, e, g in product(ps_list, e_list, g_list):
        tune_idx = tuning_dataset[ps, e, g]
        other_idx = [i for i in range(11) if i != tune_idx]
        for run_name in testing_runs:
            if run_name in ['SCoNE','SCoNE(Fro)','sHNMF']:
                matching_best_run = best[(best['params.run_name']==run_name)&(best['params.ps']==ps)&(best['params.e']==e)&(best['params.g']==g)].copy()
                assert matching_best_run.shape[0] == 1
                lambda_W, lambda_H_G, lambda_H_C = matching_best_run[['params.lambda_W','params.lambda_H_G','params.lambda_H_C']].values[0].tolist()
            else:
                lambda_W, lambda_H_G, lambda_H_C = (0,0,0)
            for init, idx in product(range(10),other_idx): # get 10 random iniitalizations of each
                combos.append((ps, e, g, lambda_W, lambda_H_G, lambda_H_C, run_name, init, idx))
    cfgs = [
        dict(
            ps=ps, e=e, dataset=idx, g=g,
            init = init,
            Z=Z if run_name not in ['sHNMF','HNMF','G-NMF','C-NMF'] else np.zeros((Z.shape[0],Z.shape[1])), W=np.random.uniform(low=0.1,high=1,size=(2504, rank)), 
            H_G=np.random.uniform(low=0.1,high=1,size=(num_markers, rank)), H_C=np.random.uniform(low=0.1,high=1,size=(100, rank)), U_G=np.random.uniform(low=0.1,high=1,size=(num_markers, Z.shape[1])),
            U_C=np.random.uniform(low=0.1,high=1,size=(100, Z.shape[1])),
            lambda_W=lambda_W, lambda_H_G=lambda_H_G, lambda_H_C=lambda_H_C,
            max_outer=50, min_outer=5, tol=1e-6, nonneg=True,
            sim_output_dir=sim_output_dir, exp_num=exp_map[ps, e, g],
            run_name=run_name,G_loss_type='kl_div' if run_name not in ['SCoNE(Fro)','C-NMF','C-CoNE'] else ('fro' if run_name=='SCoNE(Fro)' else None), 
            C_loss_type='kl_div' if run_name not in ['SCoNE(Fro)','G-NMF','G-CoNE'] else ('fro' if run_name=='SCoNE(Fro)' else None),
            artifact_dir=f"{os.path.dirname(artifact_dir)}/logs"
        )
        for i, (ps, e, g, lambda_W, lambda_H_G, lambda_H_C, run_name, init, idx) in enumerate(combos)
    ]
    mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/logs")
    for i in range(len(list(product(ps_list, e_list, g_list)))):
        exp = mlflow.set_experiment(str(i)) # set experiment id ahead of time for slurm parallelism
    jobs = executor.map_array(_call_kwargs, cfgs) 


# todo - choose over inits from best loss
