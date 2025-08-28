#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=GetAnalysisdf
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=GetAnalysisdfoutput.txt
#SBATCH --error=GetAnalysisdferrors.txt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
import pickle
from itertools import product

intermediate_file_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
root_dir = '/gpfs/commons/datasets/1000genomes'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'

results_df = []
results_df_columns = ['ps','e','af_variance_grouping','phenotypic_subgroup','tn','tn_w_pcs','fp','fp_w_pcs','fn','fn_w_pcs','tp','tp_w_pcs',
                      'accuracy','accuracy_w_pcs','precision','precision_w_pcs','recall',
                      'recall_w_pcs','f1','f1_w_pcs','avg_relative_change_w_cov']
ps_list = np.round(np.arange(0, 1.0, 0.1),1)
af_variance_grouping_list = ['superpopulation','admixture']

for ps, e, af_variance_grouping, phenotypic_subgroup in product(ps_list, [0.0,0.5,1.0], af_variance_grouping_list, range(4)):
    output_suffix = f'ps_{ps}_e_{e}_afgrouping_{af_variance_grouping}'
    with open(f"{output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    plink_results_w_wout_cov = []
    performance_metrics = {True:{},False:{}}
    for cov_included in [True,False]:
        plink_results = pd.read_csv(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_suffix}_Geno_Cov_{cov_included}.Phenotype.glm.logistic.hybrid',sep='\t')
        plink_results = plink_results[plink_results['TEST']=='ADD'].copy()
        plink_results['cov_included'] = cov_included
        plink_results_w_wout_cov.append(plink_results[['ID','OR','cov_included']])
        if (phenotypic_subgroup in range(2)) and (e>0): # first two associated with genotype, second two should have no associated genotype (and need e>0 to expect some association)
            associated_markers = simulation_metadata['markers_assoc'][phenotypic_subgroup]
            plink_results['marker_assoc'] = plink_results['ID'].isin(associated_markers) # ground truth from simulations
        else:
            associated_markers = []
            plink_results['marker_assoc'] = False
        # using significance level 5e-8, what is the accuracy of association test
        plink_results['significant'] = plink_results['P'] < 5e-8 # predicted associated

        # record accuracy, precision, recall, f1
        print(confusion_matrix(plink_results["marker_assoc"], plink_results["significant"]))
        tn, fp, fn, tp = confusion_matrix(plink_results["marker_assoc"], plink_results["significant"],labels=[False,True]).ravel()
        performance_metrics[cov_included]['tn'] = tn
        performance_metrics[cov_included]['fp'] = fp
        performance_metrics[cov_included]['fn'] = fn
        performance_metrics[cov_included]['tp'] = tp
        performance_metrics[cov_included]['accuracy'] = accuracy_score(plink_results["marker_assoc"], plink_results["significant"])
        performance_metrics[cov_included]['precision'] = precision_score(plink_results["marker_assoc"], plink_results["significant"])
        performance_metrics[cov_included]['recall'] = recall_score(plink_results["marker_assoc"], plink_results["significant"])
        performance_metrics[cov_included]['f1'] = f1_score(plink_results["marker_assoc"], plink_results["significant"])
    
    # get change in each coefficient with and without PCs
    plink_results_w_wout_cov = pd.concat(plink_results_w_wout_cov)
    wide = (plink_results_w_wout_cov.pivot(index='ID', columns='cov_included', values='OR')
            .rename(columns={False:'OR_no_cov', True:'OR_with_cov'}).reset_index())
    wide['relative_change'] = abs(wide['OR_with_cov'] - wide['OR_no_cov']) / wide['OR_no_cov']
    # record hparams
    results_df.append([ps,e,af_variance_grouping,phenotypic_subgroup,
                    performance_metrics[False]['tn'],performance_metrics[True]['tn'],
                    performance_metrics[False]['fp'],performance_metrics[True]['fp'],
                    performance_metrics[False]['fn'],performance_metrics[True]['fn'],
                    performance_metrics[False]['tp'],performance_metrics[True]['tp'],
                    performance_metrics[False]['accuracy'],performance_metrics[True]['accuracy'],
                    performance_metrics[False]['precision'],performance_metrics[True]['precision'],
                    performance_metrics[False]['recall'],performance_metrics[True]['recall'],
                    performance_metrics[False]['f1'],performance_metrics[True]['f1'],
                    wide['relative_change'].mean()]) # only get relative change in OR for known associations
        

results_df = pd.DataFrame(results_df,columns=results_df_columns)
results_df['avg_relative_change_w_cov_percent'] = results_df['avg_relative_change_w_cov']*100
results_df.to_csv('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output/results_df.csv')