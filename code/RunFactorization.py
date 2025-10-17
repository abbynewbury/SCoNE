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
r_path = '/gpfs/commons/home/anewbury/miniconda/bin/Rscript'
sys.path.append(code_dir)
import simulations.genomes1000_sim as sim_functions
import algorithms.SCoNE as SCoNE
import algorithms.RGWASWrapper as RGWASWrapper
import algorithms.MVBCWrapper as MVBCWrapper
import algorithms.MLFlowWrapper as MLFlowWrapper
import evaluation.cluster_evaluation as cluster_evaluation
import importlib
np.random.seed(42)

# PARAMETERS
# generate all combinations of e and g_ps, c_ps variables
e_list = [0.25, 0.50, 0.75, 1] 
g_ps_list = [0,0.25,0.75]
c_ps_list = [0,0.1,0.3]
dataset_list = range(11) # 11 random datasets for each combination
bfile_path=f'{sim_output_dir}/G'
af_df_filepath=admixture_filepath
rank = 3
num_markers = 100
tuning = False # to run sparsity tuning step
testing = True # to run testing step
# PARAMETERS

# read in Z
Z_df = pd.read_csv(f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q',sep='\s+',header=None)
Z = Z_df[[i for i in Z_df.columns if i!='IID']].to_numpy() 



def _call_kwargs(kw):
    return run_one_wrapper(**kw)  # expands kwargs dict

def run_one_wrapper(g_ps, c_ps, e, dataset, num_init,
                    Z,rank,num_markers,
                    lambda_W, lambda_H_G, lambda_H_C,lambda_Gloss,
                    G_loss_type, C_loss_type,
                    max_outer,min_outer,tol,nonneg,
                    sim_output_dir, exp_num,run_name,artifact_dir): 
    output_suffix = sim_functions.get_output_file_suffix(g_ps=g_ps, c_ps=c_ps, e=e, dataset=dataset)
    print(f"{sim_output_dir}/simulation_metadata_{output_suffix}.pkl")
    C_path = f'{sim_output_dir}/C_{output_suffix}.npy'
    C = np.load(C_path)
    with open(f"{sim_output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None) 
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_index = fam_df[fam_df['IID'].isin(simulation_metadata['iid_order'])].index # some samples removed due to overlap btwn subgroups, need correct length
    # read in G with num_markers
    G_path = f'{sim_output_dir}/G_{output_suffix}.raw'
    G = np.loadtxt(G_path,  usecols=range(6, num_markers+6), dtype=np.int64, skiprows=1)
    fam_df_subset = pd.read_csv(f'{sim_output_dir}/G_{output_suffix}.fam',sep='\s+',header=None)
    fam_df_subset.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    assert set(fam_df_subset['IID'].values).issubset(set(fam_df['IID'].values))
    # Note: first two columns are genetically-informed subgroups by construction
    W_true = (simulation_metadata['phenotypic_subgroups'].pivot(index='IID',columns='phenotypic_subgroup',values='subgroup')
            .reindex(simulation_metadata['iid_order']).iloc[:,:2].to_numpy().astype(int))
    ground_truth = {"W":W_true}

    if lambda_Gloss == 'ratio': # TODO: clean this up
        lambda_Gloss = C.sum()/G.sum()

    if run_name in ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)','sHNMF']:
        return MLFlowWrapper.train_with_mlflow( 
        algorithm_func=SCoNE.alternating_opt,
        artifact_dir=artifact_dir,
        run_name=run_name,
        algorithm_func_kwargs={"G":G, "C":C, "Z":Z[iid_index,:],"rank":rank, "num_init":num_init,
                               "lambda_W":lambda_W, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C,"lambda_Gloss":lambda_Gloss,
                               "method":'L-BFGS-B', "options":{'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6},  # keep scipy methods and options fixed
                               "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                               "max_outer":max_outer, "min_outer":min_outer, "tol":tol, "nonneg":nonneg},
        params={"g_ps": g_ps,"c_ps":c_ps, "e": e, "dataset": dataset, "num_init":num_init,
                "run_seed": simulation_metadata["run_seed"], "tol": tol, "max_outer": max_outer, "min_outer":min_outer,
                "lambda_W": lambda_W, "lambda_H_G": lambda_H_G, "lambda_H_C": lambda_H_C, "lambda_Gloss":lambda_Gloss, "run_name":run_name,
                "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                "method":'L-BFGS-B', "options":{'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6}},
        eval_fn=cluster_evaluation.compute_sim_metrics, 
        ground_truth=ground_truth,
        experiment_name=str(exp_num))
    elif run_name == 'MVBC':
        return  MLFlowWrapper.train_with_mlflow(algorithm_func=MVBCWrapper.MVBCWrapper,
        artifact_dir=artifact_dir,
        run_name=run_name,
        algorithm_func_kwargs={"G_path":G_path,"C_path":C_path, "rank":rank,
                    "lambda_W":lambda_W, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C, "r_path":r_path},
        params={"g_ps": g_ps,"c_ps":c_ps, "e": e, "dataset": dataset,
        "run_name":run_name, "num_init":1,
        "lambda_W":lambda_W, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C},
        eval_fn=cluster_evaluation.compute_sim_metrics, 
        ground_truth=ground_truth,
        experiment_name=str(exp_num))
    elif run_name == 'RGWAS':
        Z_path = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q'
        # save iid index
        iid_index_path = f'{sim_output_dir}/iid_index_{output_suffix}.npy'
        np.save(iid_index_path,iid_index.to_numpy())
        if len(iid_index)<500: # need to subset to only linked features 
            clinical_assoc = list(set([x for sub in simulation_metadata['clinical_assoc']['indices'].values.tolist() for x in sub]))
            C_subset = C[:,clinical_assoc] 
            np.save(f'{sim_output_dir}/C_subset_{output_suffix}.npy', C_subset)
            C_path= f'{sim_output_dir}/C_subset_{output_suffix}.npy'

            markers_assoc = list(set([x for sub in simulation_metadata['markers_assoc'].values() for x in sub]))
            with open(f'{sim_output_dir}/markers_assoc_G_subset_{output_suffix}.txt','w') as f:
                for snp in markers_assoc:
                    f.write(snp + "\n")
            plink_extract = f'''
                module load plink/1.9 && plink --bfile {sim_output_dir}/G_{output_suffix} \
                    --recode A \
                    --extract {sim_output_dir}/markers_assoc_G_subset_{output_suffix}.txt\
                    --out {sim_output_dir}/G_subset_{output_suffix}
                '''
            result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")
            G_path=f'{sim_output_dir}/G_subset_{output_suffix}.raw'

        return  MLFlowWrapper.train_with_mlflow(algorithm_func=RGWASWrapper.RGWASWrapper,
        artifact_dir=artifact_dir,
        run_name=run_name,
        algorithm_func_kwargs={"iid_index_path":iid_index_path,"r_path":r_path, "G_path":G_path,
                        "C_path":C_path, "Z_path":Z_path, "num_init":num_init,"rank":rank},
        params={"g_ps": g_ps,"c_ps":c_ps, "e": e, "dataset": dataset,
        "run_name":run_name, "num_init":num_init},
        eval_fn=cluster_evaluation.compute_sim_metrics, 
        ground_truth=ground_truth,
        experiment_name=str(exp_num))

    else: assert True == False, f"invalid run name {run_name}"

os.makedirs(artifact_dir, exist_ok=True)
os.makedirs(f"{os.path.dirname(artifact_dir)}/slurm_logs", exist_ok=True)
os.makedirs(f"{os.path.dirname(artifact_dir)}/logs_tuning", exist_ok=True)

# PRELIMINARY: set up fixed params across experiments
exp_map = {}
tuning_dataset = {}
for i, (g_ps,c_ps,e) in enumerate([(g_ps, c_ps, e) for e, g_ps in product(e_list, g_ps_list) for c_ps in ([0] if g_ps == 0 else c_ps_list)]):
    key = (g_ps,c_ps, e)
    tuning_dataset[key] = np.random.randint(0, 11)
    exp_map[key] = i
# submit as a SLURM array (adjust params as needed)
executor = submitit.AutoExecutor(folder=f"{os.path.dirname(artifact_dir)}/slurm_logs")
executor.update_parameters(
    slurm_job_name="fact-grid",
    timeout_min=300,
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
tuning_runs = ['SCoNE','SCoNE(Fro)','sHNMF','MVBC']  # all of these runs have sparsity parameters that need to be tuned 
# testing_runs = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)','sHNMF','RGWAS','MVBC'] # TODO CHANGE BACK
testing_runs = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE']
# PRELIMINARY: set up fixed params across experiments

# STEP 1: hparam tuning with 1 randomly selected dataset per experiment (and then remove it from testing)
if tuning:
    combos = [(g_ps, c_ps, e, lW, lHG, lHC, rn) for e, g_ps in product(e_list, g_ps_list) for c_ps in ([0] if g_ps == 0 else c_ps_list) for (lW, lHG, lHC, rn) in product([0,0.5,1],[0,0.5,1],[0,0.5,1], tuning_runs)]


    cfgs = [
        dict(
            g_ps=g_ps, c_ps=c_ps, e=e, dataset=tuning_dataset[g_ps,c_ps, e],
            num_init = 10, num_markers=num_markers,
            Z=Z if run_name != 'sHNMF' else np.zeros((Z.shape[0],Z.shape[1])), rank=rank,
            lambda_W=lambda_W, lambda_H_G=lambda_H_G, lambda_H_C=lambda_H_C,lambda_Gloss=1,
            max_outer=50, min_outer=5, tol=1e-6, nonneg=True,
            sim_output_dir=sim_output_dir, exp_num=exp_map[g_ps,c_ps, e],
            run_name=run_name,G_loss_type='kl_div' if run_name!='SCoNE(Fro)' else 'fro', C_loss_type='kl_div' if run_name!='SCoNE(Fro)' else 'fro',
            artifact_dir=f"{os.path.dirname(artifact_dir)}/logs_tuning"
        )
        for i, (g_ps,c_ps, e, lambda_W, lambda_H_G, lambda_H_C, run_name) in enumerate(combos)
    ]
    mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/logs_tuning")
    for i in exp_map.values():
        exp = mlflow.set_experiment(str(i)) # set experiment id ahead of time for slurm parallelism
    jobs = executor.map_array(_call_kwargs, cfgs)

    # Block until every array task has completed (and raise if any failed)
    t0 = datetime.now(ZoneInfo("America/New_York"))
    _ = [j.result() for j in jobs]
    t1 = datetime.now(ZoneInfo("America/New_York"))
    print(f"[{t1:%Y-%m-%d %H:%M:%S %Z}] now running… elapsed={t1 - t0}", flush=True)

if testing:
    if any(x in testing_runs for x in tuning_runs):
        # STEP 2: find optimal sparsity parameters for each method
        mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/logs_tuning")
        client = MlflowClient()
        # Get all experiments
        all_runs = []
        for exp in client.search_experiments():
            df = mlflow.search_runs([exp.experiment_id])
            all_runs.append(df)
        all_runs = pd.concat(all_runs, ignore_index=True)
        all_runs.columns=[i.replace('params.','').replace('metrics.','') for i in all_runs.columns]
        all_runs[['e','g_ps','c_ps','lambda_W','lambda_H_G','lambda_H_C']]  = all_runs[['e','g_ps','c_ps','lambda_W','lambda_H_G','lambda_H_C']].astype("float64") 
        idx = all_runs.groupby(['run_name', 'g_ps','c_ps', 'e'])["G_plus_C_loss"].idxmin()
        best = (
            all_runs.loc[idx, all_runs.columns]
            .reset_index(drop=True)
        )

    # STEP 3: testing/comparison over 10 other datasets for each experiment
    # submit as a SLURM array (adjust params as needed)
    # build combos excluding the tuning dataset 
    combos = []
    for (g_ps,e) in product(g_ps_list, e_list):
        for c_ps in ([0] if g_ps == 0 else c_ps_list):
            tune_idx = tuning_dataset[g_ps, c_ps, e]
            other_idx = [i for i in dataset_list if i != tune_idx]
            for run_name in testing_runs:
                if run_name in tuning_runs:
                    matching_best_run = best[(best['run_name']==run_name)&(best['g_ps']==g_ps)&(best['c_ps']==c_ps)&(best['e']==e)].copy()
                    assert matching_best_run.shape[0] == 1
                    lambda_W, lambda_H_G, lambda_H_C = matching_best_run[['lambda_W','lambda_H_G','lambda_H_C']].values[0].tolist()
                else:
                    lambda_W, lambda_H_G, lambda_H_C = (0,0,0)
                for idx in other_idx: # get 10 random initalizations of each
                    combos.append((g_ps, c_ps, e, lambda_W, lambda_H_G, lambda_H_C, run_name, idx))
    cfgs = [
        dict(
            g_ps=g_ps, c_ps=c_ps, e=e, dataset=idx,
            num_init = 10, num_markers=num_markers,
            Z=Z if run_name not in ['sHNMF','HNMF','G-NMF','C-NMF'] else np.zeros((Z.shape[0],Z.shape[1])), rank=rank,
            lambda_W=lambda_W, lambda_H_G=lambda_H_G, lambda_H_C=lambda_H_C, lambda_Gloss=1,
            max_outer=50, min_outer=5, tol=1e-6, nonneg=True,
            sim_output_dir=sim_output_dir, exp_num=exp_map[g_ps, c_ps, e],
            run_name=run_name,G_loss_type='kl_div' if run_name not in ['SCoNE(Fro)','C-NMF','C-CoNE'] else ('fro' if run_name=='SCoNE(Fro)' else None), 
            C_loss_type='kl_div' if run_name not in ['SCoNE(Fro)','G-NMF','G-CoNE'] else ('fro' if run_name=='SCoNE(Fro)' else None),
            artifact_dir=f"{os.path.dirname(artifact_dir)}/new_logs" # TODO: change back
        )
        for i, (g_ps, c_ps, e, lambda_W, lambda_H_G, lambda_H_C, run_name, idx) in enumerate(combos)
    ]
    mlflow.set_tracking_uri("file:" + f"{os.path.dirname(artifact_dir)}/new_logs") # TODO: change back
    for i in exp_map.values():
        exp = mlflow.set_experiment(str(i)) # set experiment id ahead of time for slurm parallelism
    jobs = executor.map_array(_call_kwargs, cfgs)


