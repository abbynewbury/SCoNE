#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=SimulateData
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
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
intermediate_file_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
root_dir = '/gpfs/commons/datasets/1000genomes'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
maf_by_superpop_filepath = f'{intermediate_file_dir}/maf_by_superpop'
admixture_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5'
map_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map'
code_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code'
# DEFINE PATHS

sys.path.append(code_dir)
from simulations.genomes1000_sim import *


# PARAMETERS
generate_sim = True
evaluate_sim = True
# generate all combinations of e and ps variables
ps_list = [True,False]
e_list = [0.00, 0.25, 0.50, 0.75, 1]
num_markers_assoc_list = [100,2000]
init_list = range(100) # 100 random initializations for each combination
# PARAMETERS


# GENERATE SIMULATED DATA

if generate_sim:
    # RUN FILE SETUP
    # generate genetic bfile (outputs to {output_dir}/G)
    prep_1000genomes_bed_file(root_dir=root_dir, output=f'{output_dir}/G',subset_test=True) # if subset_test is true - only use 10k snps for faster processing
    #maf_by_superpop = calculate_maf_by_superpop(igsr_samples_filepath,intermediate_file_dir,bfile_path=f'{output_dir}/G',output=maf_by_superpop_filepath)
    # get pcs - for later gwas evaluation
    result = subprocess.run(f'module unload plink && module load flashpca && cd {output_dir} &&  flashpca --bfile {output_dir}/G --ndim 20', shell=True, capture_output=True, text=True, executable='/bin/bash')
    # RUN FILE SETUP


    # run with 2000 or 100 associated markers, 100 random initializations each
    run_one = partial(
        sun_generate_sim_data,
        bfile_path=f'{output_dir}/G', af_df_filepath=admixture_filepath,
        map_filepath = map_filepath,
        intermediate_file_dir=intermediate_file_dir,
        output_dir=output_dir,
        extra_subgroups_size=200,
        M=100,
        num_clinical_assoc=10
    )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            ps=ps, e=e, num_markers_assoc=num_markers_assoc, intermediate_file_suffix=get_output_file_suffix(ps,e,init,num_markers_assoc), output_file_suffix=get_output_file_suffix(ps,e,init,num_markers_assoc)) 
            for ps, e, init, num_markers_assoc in product(ps_list, e_list, init_list, num_markers_assoc_list)
        )


# EVALUATE SIMULATED DATA

if evaluate_sim:
# VISUAL EVALUATION (UMAP)
    # # For ps
    # generate_umap_plot(mode='ps', var_list=ps_list, color_col='Superpopulation code', 
    #                         color_label='Superpopulation', output_dir=output_dir,igsr_samples_filepath=igsr_samples_filepath)

    # # For e
    # generate_umap_plot(mode='e', var_list=e_list, color_col='subgroup_value', 
    #                         color_label='Genetic Subgroup', output_dir=output_dir)


# QUANTITATIVE EVALUATION (GWAS)

    # write to covariate file
    igsr_samples = read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/G')
    pcs = pd.read_csv(f'{output_dir}/pcs.txt',sep='\t')
    covar = pcs.merge(igsr_samples[['IID','Sex']], on='IID',how='inner')
    # probably cleaner way to do this
    covar[['FID','IID']+[f'PC{i}' for i in range(1,6)]+['Sex']].set_index('FID').to_csv(f'{intermediate_file_dir}/COVARIATE_FILE')
    covar[['FID','IID']+['Sex']].set_index('FID').to_csv(f'{intermediate_file_dir}/COVARIATE_FILE_NOPS')

    # run gwas phenotypic subgroup ~ genotypes + age + (pcs?)
    run_one = partial(
    run_phenotypicsubgroup_gwas,
    intermediate_file_dir=intermediate_file_dir,
    output_dir=output_dir
    )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            output_file_suffix=get_output_file_suffix(ps,e,init,num_markers_assoc),ps=ps, e=e, cov_included=cov_included, phenotypic_subgroup=phenotypic_subgroup) 
            for ps, e, init, num_markers_assoc, cov_included, phenotypic_subgroup  in product(ps_list, [0.50], init_list, num_markers_assoc_list, [True,False],range(4))
        )