#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=SimulateData
#SBATCH --nodes=1
#SBATCH --mem=30G
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
evaluate_sim = True 
run_gwas = False  # only will run if run_gwas=True AND evaluate_sim=True 
# generate all combinations of e and ps variables
ps_list = [True,False]
e_list = [0.25, 0.50, 0.75, 1]
g_list = [10] # g represents the number of linked markers
dataset_list = range(11) # 11 random datasets for each combination
# PARAMETERS


# GENERATE SIMULATED DATA

if generate_sim:
    # RUN FILE SETUP
    # generate genetic bfile (outputs to {output_dir}/G)
    prep_1000genomes_bed_file(root_dir=root_dir, output=f'{output_dir}/G',subset_test=False) # if subset_test is true - only use 10k snps for faster processing
    # calculate maf by superpopulation
    maf_by_superpop = calculate_maf_by_superpop(igsr_samples_filepath,intermediate_plink_dir,bfile_path=f'{output_dir}/G',output=maf_by_superpop_filepath)
    # RUN FILE SETUP
    
    combos = list(product(ps_list, e_list, dataset_list, g_list))
    child_ss = np.random.SeedSequence().spawn(len(combos)) 
    run_seeds = [int(np.random.default_rng(ss).integers(1, 2**31 - 1)) for ss in child_ss] # for reproducible randomness
    # run with 500 or 100 associated markers, 11 random datasets each
    run_one = partial(
        sun_generate_sim_data,
        bfile_path=f'{output_dir}/G', maf_by_superpop_filepath=f'{intermediate_plink_dir}/maf_by_superpop.frq.strat',
        igsr_samples_filepath = igsr_samples_filepath,
        intermediate_file_dir=intermediate_plink_dir,
        output_dir=output_dir,
        extra_subgroups_size=200,
        M=100,
        num_clinical_assoc=10,
        num_markers=100
    )

    results = Parallel(n_jobs=-1)(
        delayed(run_one)(
            ps=ps, e=e, g=g, intermediate_file_suffix=get_output_file_suffix(ps,e,dataset,g),
              output_file_suffix=get_output_file_suffix(ps,e,dataset,g),run_seed=run_seeds[i]) 
            for i, (ps, e, dataset, g) in enumerate(combos)
        )


# EVALUATE SIMULATED DATA

if evaluate_sim:
# VISUAL EVALUATION (UMAP)
    # For ps
    generate_umap_plot(mode='ps', var_list=ps_list, color_col='Superpopulation code', 
                            color_label='Superpopulation', output_dir=output_dir,
                            graph_dir=graph_dir,igsr_samples_filepath=igsr_samples_filepath)

    # For e
    generate_umap_plot(mode='e', var_list=e_list, color_col='subgroup_value', 
                            color_label='Genetic Subgroup', output_dir=output_dir,graph_dir=graph_dir)


# QUANTITATIVE EVALUATION (GWAS)
    if run_gwas:
        # get pcs 
        get_genetic_pcs(map_ped_filepath=f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05',
                    output_dir=output_dir, ndim=20)

        # write to covariate file
        igsr_samples = read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/G')
        pcs = pd.read_csv(f'{output_dir}/pcs.txt',sep='\t')
        covar = pcs.merge(igsr_samples[['IID','Sex']], on='IID',how='inner')
        # probably cleaner way to do this
        covar[['FID','IID']+[f'PC{i}' for i in range(1,11)]+['Sex']].set_index('FID').to_csv(f'{output_dir}/COVARIATE_FILE')


        # run gwas phenotypic subgroup ~ genotypes + age + pcs (plink w/out covs, plink w/ covs, saige w/ covs)
        run_one = partial(
        run_phenotypicsubgroup_gwas,
        intermediate_plink_dir=intermediate_plink_dir,
        intermediate_saige_dir=intermediate_saige_dir,
        output_dir=output_dir
        )

        results = Parallel(n_jobs=24)(
            delayed(run_one)(
                output_file_suffix=get_output_file_suffix(ps,e,dataset,g), phenotypic_subgroup=phenotypic_subgroup) 
                for ps, e, dataset, g,  phenotypic_subgroup  in product(ps_list, [0.50,1], dataset_list, g_list,range(4))
            )
        
    # make plots evaluating gwas
    combos = list(product(ps_list, [0.50,1], dataset_list, g_list, range(4)))
    results_df_indiv = Parallel(n_jobs=-1)(
        delayed(evaluate_gwas)(
            output_dir=output_dir,
            ps=ps, e=e, dataset=dataset, g=g, 
            phenotypic_subgroup=phenotypic_subgroup, sig_level=5e-8
        )
        for (ps, e, dataset, g, phenotypic_subgroup) in combos
    )

    results_df = pd.concat(results_df_indiv)

    # plot of mean abs(beta) for linked markers
    p = (ggplot(results_df[results_df['assoc_test']!='LR'], aes(x='ps', y='avg_abs_beta_assoc',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(12,8))
    + facet_grid('assoc_test ~ g')
    + labs(title=r'Mean abs($\beta$) of Linked Markers',y=r'Mean abs($\beta$)',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_avgbeta_linked.png", dpi=300)

    p = (ggplot(results_df[results_df['assoc_test']=='LR'], aes(x='ps', y='avg_abs_beta_assoc',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(12,4))
    + facet_grid('assoc_test ~ g')
    + labs(title=r'Mean abs($\beta$) of Linked Markers',y=r'Mean abs($\beta$)',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_avgbeta_linked_LR.png", dpi=300)

    # plot representing confounding
    results_df['subgroup_type'] = results_df['phenotypic_subgroup'].apply(lambda x: 'genetic link subgroup' if x in range(2) else 'random subgroup')
    p = (ggplot(results_df[results_df['assoc_test']!='LR'], aes(x='ps', y='prop_rel_change_gt10',fill='factor(e)'))
        + geom_boxplot()
        + theme_minimal()
        + theme(figure_size=(12,8))
        + facet_grid('assoc_test ~ subgroup_type')
        + labs(title=r'% of All Markers w/ relative change in $\beta$ > $ \pm $ 10% vs. simple LR',y=r'Proportion',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_confounding_pctchange.png", dpi=300)

    # performance metrics
    p = (ggplot(results_df, aes(x='ps', y='accuracy',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(16,8))
    + facet_grid('assoc_test ~ subgroup_type')
    + labs(title=r'Accuracy in identifying linked markers',y=r'Accuracy',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_accuracy.png", dpi=300)

    p = (ggplot(results_df, aes(x='ps', y='recall',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(16,8))
    + facet_grid('assoc_test ~ subgroup_type')
    + labs(title=r'Sensitivity in identifying linked markers',y=r'Sensitivity',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_sensitivity.png", dpi=300)

    p = (ggplot(results_df, aes(x='ps', y='specificity',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(16,8))
    + facet_grid('assoc_test ~ subgroup_type')
    + labs(title=r'Specificity in identifying linked markers',y=r'Specificity',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_specificity.png", dpi=300)

    results_df['pred_pos'] = results_df['tp'] + results_df['fp'] 
    p = (ggplot(results_df, aes(x='ps', y='pred_pos',fill='factor(e)'))
    + geom_boxplot()
    + theme_minimal()
    + theme(figure_size=(16,8))
    + facet_grid('assoc_test ~ subgroup_type',scales='free_y')
    + labs(title=r'# GWAS Hits',y=r'# GWAS Hits',x=r'$p_s$')) 
    p.save(f"{graph_dir}/SimValidation_numgwashits.png", dpi=300)
    