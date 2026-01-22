#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=SimulateData
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=SimulateDataoutput.txt
#SBATCH --error=SimulateDataerrors.txt

import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
import sys
from itertools import product
from joblib import Parallel, delayed
from functools import partial

# DEFINE PATHS
intermediate_plink_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
intermediate_saige_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_saige'
root_dir = '/gpfs/commons/datasets/1000genomes'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
graph_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/output'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
maf_by_superpop_filepath = f'{intermediate_plink_dir}/maf_by_superpop'
admixture_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5'
map_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map'
code_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code'
# DEFINE PATHS

sys.path.append(code_dir)
from simulations.genomes1000_sim import *


# PARAMETERS
generate_sim = True
evaluate_sim = False 
run_gwas = False  # only will run if run_gwas=True AND evaluate_sim=True 
# generate all combinations for 5 scenarios
e_list = [0.6,0.8,1] # TODO [0.4,0.6,0.8,1]
g_ps_list = [0,0.25] # TODO [0,0.25,0.75]
c_ps_list = [0,0.1,1] # TODO [0,0.1,0.5,1,2,3,3.5,4,10] 
dataset_list = range(11) # 21 random datasets for each combination 
#combos = [(e, g_ps, c_ps, d) for e, g_ps in product(e_list, g_ps_list) for c_ps in ([0] if g_ps == 0 else c_ps_list) for d in dataset_list]
combos = [(e, g_ps, c_ps, d) for e, g_ps, c_ps, d in product(e_list, g_ps_list,c_ps_list,dataset_list)]
# PARAMETERS


# GENERATE SIMULATED DATA

if generate_sim:
    # # RUN FILE SETUP
    # # generate SNP-wise bfile and gene-wise burden matrix (outputs to {output_dir}/G)
    # prep_1000genomes_bed_file(root_dir=root_dir, output=f'{output_dir}/G',intermediate_dir=intermediate_plink_dir)
    # # # RUN FILE SETUP
    
    child_ss = np.random.SeedSequence().spawn(len(combos))
    run_seeds = [int(np.random.default_rng(ss).integers(1, 2**31 - 1)) for ss in child_ss] # for reproducible randomness
    # 11 random datasets each
    run_one = partial(
        sun_generate_sim_data,
        G_path=f'{output_dir}/G', 
        igsr_samples_filepath = igsr_samples_filepath,
        output_dir=output_dir
    )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            g_ps=g_ps, c_ps=c_ps, e=e, 
              output_file_suffix=get_output_file_suffix(e,g_ps,c_ps,dataset),run_seed=run_seeds[i], g=1 if g_ps==0 else 10, M=10 if g_ps==0 else 60, 
                num_clinical_assoc=1 if g_ps==0 else 10, num_genes=10 if g_ps==0 else 60) # g_ps = 0 implies EUR individuals only (so smaller sample size)
            for i, (e,g_ps,c_ps,dataset) in enumerate(combos)
        )
