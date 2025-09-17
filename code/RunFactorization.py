#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=RunFactorization
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
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


intermediate_plink_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
intermediate_saige_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_saige'
root_dir = '/gpfs/commons/datasets/1000genomes'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
admixture_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5'
map_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map'
code_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code'
artifact_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/output/logs'

# PARAMETERS
generate_sim = True
evaluate_sim = True
# generate all combinations of e and ps variables
ps_list = [True,False]
e_list = [0.25,0.75,0.5,1] 
num_markers_assoc_list = [100,500]
init_list = range(1) # 100 random initializations for each combination, TODO: change to 100
bfile_path=f'{output_dir}/G'
af_df_filepath=admixture_filepath
# PARAMETERS

sys.path.append(code_dir)
import simulations.genomes1000_sim as sim_functions
import algorithms.JointMF as JointMF
import algorithms.JointMF_AAO as JointMF_AAO
import algorithms.MLFlowWrapper as MLFlowWrapper
import evaluation.cluster_evaluation as cluster_evaluation
import importlib

# read in G, Z
igsr_samples = sim_functions.read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/G')
igsr_samples['Sex'] = igsr_samples['Sex'].map({'female': 0, 'male': 1})

admixture = pd.read_csv(f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q',sep='\s+',header=None)
fam_df = pd.read_csv(f'{output_dir}/G.fam',sep='\s+',header=None)
fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
admixture['IID'] = fam_df['IID'].values
Z_df = admixture.merge(igsr_samples[['IID','Sex']], on='IID',how='inner')
Z = Z_df[[i for i in Z_df.columns if i!='IID']].to_numpy()

G = np.loadtxt(f'{output_dir}/G.raw',  usecols=range(6, 10000+6), dtype=np.int8, skiprows=1)
G = (G>0).astype(np.int8)
# assert that bim_df and G order is the same
bfile_path = f'{output_dir}/G'
bim_df = pd.read_csv(f'{bfile_path}.bim',sep='\s+',header=None,names=['CHR','SNP','CM','POS','A1','A2']).reset_index(drop=True)
G_columns = [i.split('_')[0] for i in pd.read_csv(f'{bfile_path}.raw',sep='\s+',usecols=range(6, 10000+6),nrows=1).columns]
assert all(bim_df['SNP'].values==G_columns)

# run factorization
rank = 3
lambda_W, lambda_H_G, lambda_H_C = (0,0,0) # TODO, grid search over these
W = np.random.random((G.shape[0], rank))*1e-2+1e-6 
H_G = np.random.random((G.shape[1], rank))*1e-2+1e-6
H_C = np.random.random((100, rank))*1e-2+1e-6 # 100 clinical vars
U_G = np.random.random((G.shape[1], Z.shape[1]))*1e-2+1e-6
U_C = np.random.random((100, Z.shape[1]))*1e-2+1e-6

combos = list(product(ps_list, e_list, init_list, num_markers_assoc_list))
all_results = []

def run_one_wrapper(run_one, ps, e, init, num_markers_assoc, exp_num, Z_included): # TODO: really need to clean up code, very convoluted
    output_suffix = sim_functions.get_output_file_suffix(ps=ps, e=e, init=init, num_markers_assoc=num_markers_assoc)
    C = np.load(f'{output_dir}/C_{output_suffix}.npy')
    with open(f"{output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    ground_truth = {"W":(simulation_metadata['phenotypic_subgroups'].pivot(index='IID',columns='phenotypic_subgroup',values='subgroup')
          .reindex(simulation_metadata['iid_order']).to_numpy().astype(int))}
    if Z_included:
        Z_value = Z.copy()
    else:
        Z_value = np.zeros((Z.shape[0],Z.shape[1]))
    return run_one(
        algorithm_func_inputs=(G, C, Z_value, W, H_G, H_C, U_G, U_C,
                               lambda_W, lambda_H_G, lambda_H_C,
                               'L-BFGS-B', {'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6},
                               30, 1e-4, True),
        params={"ps": ps, "e": e, "init": init, "num_markers_assoc": num_markers_assoc,
                "run_seed": simulation_metadata["run_seed"], "tol": 1e-4, "max_outer": 30,
                "lambda_W": lambda_W, "lambda_H_G": lambda_H_G, "lambda_H_C": lambda_H_C},
        eval_fn=cluster_evaluation.compute_sim_metrics,
        ground_truth=ground_truth,
        experiment_name=str(exp_num)
    )

os.makedirs(artifact_dir, exist_ok=True)

# each unique dataset has its own experiment name

run_one = partial( 
    MLFlowWrapper.train_with_mlflow,
    algorithm_func=JointMF.alternating_opt,
    artifact_dir=artifact_dir,
    run_name='ours'
)


results = Parallel(n_jobs=-1)(
    delayed(run_one_wrapper)(run_one, ps, e, init, num_markers_assoc, i, Z_included=True)
        for i, (ps, e, init, num_markers_assoc) in enumerate(combos)
    )

all_results.extend(results)

run_one = partial( 
    MLFlowWrapper.train_with_mlflow,
    algorithm_func=JointMF.alternating_opt,
    artifact_dir=artifact_dir,
    run_name='hnmf'
)


results = Parallel(n_jobs=-1)(
    delayed(run_one_wrapper)(run_one, ps, e, init, num_markers_assoc, i, Z_included=False)
        for i, (ps, e, init, num_markers_assoc) in enumerate(combos)
    )

all_results.extend(results)

all_results = pd.DataFrame(all_results)
all_results.to_csv(f'{output_dir}/eval_results.csv')