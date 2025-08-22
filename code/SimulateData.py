import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
from simulations.genomes1000_sim import *
from itertools import product
from joblib import Parallel, delayed
from functools import partial

# DEFINE PATHS
intermediate_file_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
root_dir = '/gpfs/commons/datasets/1000genomes'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
maf_by_superpop_filepath = f'{intermediate_file_dir}/maf_by_superpop'
# DEFINE PATHS

# PARAMETERS
generate_sim = False
evaluate_sim = True
# generate all combinations of e and ps variables
ps_list = np.round(np.arange(0, 1.0, 0.1),1)
e_list = np.round(np.arange(0, 1.1, 0.1),1)
# PARAMETERS


# GENERATE SIMULATED DATA

if generate_sim:
    #  RUN FILE SETUP
    # generate genetic bfile (outputs to {output_dir}/X)
    prep_1000genomes_bed_file(root_dir=root_dir, output=f'{output_dir}/X')
    maf_by_superpop = calculate_maf_by_superpop(igsr_samples_filepath,intermediate_file_dir,bfile_path=f'{output_dir}/X',output=maf_by_superpop_filepath)

    run_one = partial(
        sun_generate_sim_data,
        bfile_path=f'{output_dir}/X', maf_by_superpop_filepath=f'{intermediate_file_dir}/maf_by_superpop',
        intermediate_file_dir=intermediate_file_dir,
        output_dir=output_dir,
        num_markers_assoc=2000,
        extra_subgroups_size=200,
        M=100,
        num_clinical_assoc=10
    )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            ps=ps, e=e, intermediate_file_suffix=f'ps_{ps}_e_{e}', output_file_suffix=f'ps_{ps}_e_{e}') 
            for ps, e in product(ps_list, e_list)
        )


# EVALUATE SIMULATED DATA

if evaluate_sim:
# VISUAL EVALUATION (UMAP)

    # For ps
    generate_umap_plot(mode='ps', var_list=ps_list, color_col='Superpopulation name',
                            color_label='Superpopulation', output_dir=output_dir,igsr_samples_filepath=igsr_samples_filepath)

    # For e
    generate_umap_plot(mode='e', var_list=e_list, color_col='subgroup_value', 
                            color_label='Genetic Subgroup', output_dir=output_dir)


# QUANTITATIVE EVALUATION (GWAS)

# For ps