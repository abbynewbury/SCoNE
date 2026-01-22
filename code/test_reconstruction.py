#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=test_reconstruction
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=test_reconstruction.txt
#SBATCH --error=test_reconstruction.txt

import pandas as pd
from itertools import product
import submitit
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment
from plotnine import *
import sys
import json
from sklearn.model_selection import train_test_split

colors_dict = {
"SCoNE":"#2f4b7c",
"MVBC":"#665191",
"RGWAS":"#a05195",  
"C-NMF":"#d45087",  
"C-CoNE":"#f95d6a",  
"G-NMF":"#ff7c43",  
"G-CoNE":"#ffa600",  
"HNMF":"#2ca02c",  
"SCoNE(Fro)":"#1f77b4",
"CoNE":"#17becf"
}

# make tmp dir
root_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno'
code_dir = f'{root_dir}/code'
tmp_folder = f"{root_dir}/output/tmp"
os.makedirs(tmp_folder, exist_ok=True)
sys.path.append(code_dir)
import algorithms.SCoNE as SCoNE
import algorithms.MVBCWrapper as MVBCWrapper
import algorithms.RGWASWrapper as RGWASWrapper
import evaluation.cluster_evaluation as cluster_evaluation
import evaluation.reconstruction_evaluation as reconstruction_evaluation
from simulate_data import *


# CODE TO RUN ALL COMPARISON METHODS (AND PARALLELIZE)
def _call_kwargs(kw):
    return run_one_wrapper(**kw)  # expands kwargs dict

def run_one_wrapper(run_name,G,C,Z,lambda_W,lambda_H_G,lambda_H_C,lambda_Gloss,G_loss_type,C_loss_type,
                    G_path,C_path,Z_path,sim,variable_name,variable,rank=3,init_name=1,num_init=1,write=False):
    results = []
    results_columns = ['run_name','factor_matrix','init','sim','tuning_loss']

    if run_name in ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)']:
        # per Kim and Park, set alpha to the maximum element of G or C squared
        if run_name in ['SCoNE','SCoNE(Fro)']:
            alpha = max(G.max(),C.max())**2
        else:
            alpha = 0
        assert lambda_Gloss == 1, "right now code recording G loss doesn't rescale for lambda"
        algorithm_func_kwargs={"G":G, "C":C, "Z":Z,"rank":rank, "num_init":num_init, 
                        "alpha":alpha, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C,"lambda_Gloss":lambda_Gloss,
                        "max_inner":50, "rho":0.1, "sigma":1e-4, "inner_ftol":1e-4,
                        "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                        "max_outer":500, "min_outer":5, "tol":1e-6,"post_hoc_rescale":False}
        factor_matrices, loss_function = SCoNE.SCoNE_parallel(**algorithm_func_kwargs)
        if write is True and run_name=='SCoNE': # save so can visualize the loss in tuning over different lambda params
            with open(f"/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/output/models/loss_function_SCoNE_{variable_name}_{variable}_init_{init_name}.json", "w") as f: # TODO: remove
                json.dump(loss_function, f)
        G_loss = loss_function['G_loss'][-1]
        C_loss = loss_function['C_loss'][-1]
        tuning_loss = G_loss + C_loss
        if not 'C-' in run_name:
            rel_error_G = G_loss/reconstruction_evaluation.kl_rel_error_denom(G)
        else: rel_error_G = None
        if not 'G-' in run_name:
            rel_error_C = C_loss/reconstruction_evaluation.kl_rel_error_denom(C)
        else: rel_error_C = None

    elif run_name == 'MVBC':
        algorithm_func_kwargs = {"G_path":G_path,"C_path":C_path, "rank":rank,
                    "lambda_W":lambda_W, "lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C, "r_path":'/gpfs/commons/home/anewbury/miniconda/bin/Rscript'}
        factor_matrices, loss_function = MVBCWrapper.MVBCWrapper(**algorithm_func_kwargs)
        if factor_matrices is False: # run failed for reasons specified in MVBCWrapper
            return pd.DataFrame() 
        tuning_loss = loss_function['G_plus_C_loss'][-1]
        rel_error_G = None
        rel_error_C = None
    elif run_name == 'RGWAS':
        algorithm_func_kwargs = {"r_path":'/gpfs/commons/home/anewbury/miniconda/bin/Rscript', "G_path":G_path,
                        "C_path":C_path, "Z_path":Z_path, "num_init":1,"rank":rank}
        factor_matrices, loss_function = RGWASWrapper.RGWASWrapper(**algorithm_func_kwargs)
        if factor_matrices is False: # run failed for reasons specified in RGWASWrapper
            return pd.DataFrame() 
        tuning_loss = -loss_function['ll'][-1]
        rel_error_G = None
        rel_error_C = None
    else: assert True == False, f"invalid run name {run_name}"
    for k,v in factor_matrices.items():
        if k=="W":
            results.append([run_name,"W_C",init_name,reconstruction_evaluation.best_permutation_similarity(sim["W_C"],v),tuning_loss])
            results.append([run_name,"W_G",init_name,reconstruction_evaluation.best_permutation_similarity(sim["W_G"],v),tuning_loss])
        else:
            results.append([run_name,k,init_name,reconstruction_evaluation.best_permutation_similarity(sim[k],v),tuning_loss])
    results = pd.DataFrame(results,columns=results_columns)
    results[variable_name] = variable
    results['rel_error_G'] = rel_error_G
    results['rel_error_C'] = rel_error_C
    results['lambda_W'] = lambda_W
    results['lambda_H_G'] = lambda_H_G
    results['lambda_H_C'] = lambda_H_C
    results['rank'] = rank
    return results

def get_summary_df(df,grouping_vars,value_var):
    summary = (
        df.groupby(grouping_vars)[value_var]
            .agg(n='count', mean='mean', std='std').reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["n"])         # normal approx 95% CI
    summary["ymin"] = summary["mean"] - summary["ci95"]
    summary["ymax"] = summary["mean"] + summary["ci95"]
    return summary

def run_one(variable_name, variable_range, output_dir):
    testing_runs = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)','RGWAS','MVBC']
    tuning_runs = ['SCoNE' ,'SCoNE(Fro)','MVBC']
    all_testing_results = []
    cfgs = []
    if variable_name != 'sparsity': # want tuning only to inform testing
        for variable in variable_range:
            # SIMUALTE DATA 
            sim_kwargs = {"n":500,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":0} 
            sim_kwargs[variable_name] = variable
            sim = simulate_views(**sim_kwargs) 
            # write G,C,Z to paths
            np.save(f'{tmp_folder}/G_{variable_name}_{variable}',sim["G"])
            np.save(f'{tmp_folder}/C_{variable_name}_{variable}',sim["C"])
            np.save(f'{tmp_folder}/Z_{variable_name}_{variable}',sim["Z"])
            # TUNING
            lambda_options = [0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]  # TODO: if I ever want to do product (combos) this will become redundant
            for lambda_val in lambda_options:
                for run_name in tuning_runs:
                    algorithm_kwargs = {"run_name":run_name,"G":sim["G"] if not 'C-' in run_name else None,
                                    "C":sim["C"] if not 'G-' in run_name else None,
                                    "Z":sim["Z"] if not 'NMF' in run_name else np.zeros((sim["Z"].shape[0],sim["Z"].shape[1])),
                                    "lambda_W":lambda_val,"lambda_H_G":lambda_val,"lambda_H_C":lambda_val,"lambda_Gloss":1,
                                    "G_loss_type":None if 'C-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                    "C_loss_type":None if 'G-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                    "sim":sim,"rank":3,"init_name":None, "num_init":10,
                                    "G_path":f'{tmp_folder}/G_{variable_name}_{variable}.npy',
                                    "C_path":f'{tmp_folder}/C_{variable_name}_{variable}.npy',
                                    "Z_path":f'{tmp_folder}/Z_{variable_name}_{variable}.npy',"variable_name":variable_name,"variable":variable,"write":False}
                    cfgs.append(algorithm_kwargs)
        with ProcessPoolExecutor() as pool:
            results_list = list(pool.map(_call_kwargs, cfgs))
        tuning_results = pd.concat(results_list, ignore_index=True)
        idx = tuning_results.groupby(['run_name'])["tuning_loss"].idxmin()
        best = (
            tuning_results.loc[idx, tuning_results.columns]
            .reset_index(drop=True)
        )

    for variable in variable_range:
        # testing set
        sim_kwargs = {"n":500,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":1} 
        sim_kwargs[variable_name] = variable
        sim = simulate_views(**sim_kwargs) 

        # write G,C,Z to paths
        np.save(f'{tmp_folder}/G_{variable_name}_{variable}',sim["G"])
        np.save(f'{tmp_folder}/C_{variable_name}_{variable}',sim["C"])
        np.save(f'{tmp_folder}/Z_{variable_name}_{variable}',sim["Z"])
        cfgs = []
        for init in range(10): 
            for run_name in testing_runs:
                if variable_name != 'sparsity':
                    if run_name in tuning_runs:
                        assert best[(best['run_name']==run_name)].shape[0] == 1
                        lambda_val = best[(best['run_name']==run_name)]['lambda_H_G'].values[0].tolist()
                    else:
                        lambda_val = 0
                    cfgs.append({"run_name":run_name,"G":sim["G"] if not 'C-' in run_name else None,
                                    "C":sim["C"] if not 'G-' in run_name else None,
                                    "Z":sim["Z"] if not 'NMF' in run_name else np.zeros((sim["Z"].shape[0],sim["Z"].shape[1])),
                                    "lambda_W":lambda_val,"lambda_H_G":lambda_val,"lambda_H_C":lambda_val,"lambda_Gloss":1,
                                    "G_loss_type":None if 'C-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                    "C_loss_type":None if 'G-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                    "sim":sim,"rank":3,"init_name":init,"num_init":1,
                                    "G_path":f'{tmp_folder}/G_{variable_name}_{variable}.npy',
                                    "C_path":f'{tmp_folder}/C_{variable_name}_{variable}.npy',
                                    "Z_path":f'{tmp_folder}/Z_{variable_name}_{variable}.npy',"variable_name":variable_name,"variable":variable,"write":True})
                else:
                    # try with other ranks to demonstrate the benefits of sparse params
                    for rank in range(2,7):
                        if run_name in tuning_runs:
                            lambda_options = [0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100] 
                            for lambda_val in lambda_options:
                                cfgs.append({"run_name":run_name,"G":sim["G"] if not 'C-' in run_name else None,
                                            "C":sim["C"] if not 'G-' in run_name else None,
                                            "Z":sim["Z"] if not 'NMF' in run_name else np.zeros((sim["Z"].shape[0],sim["Z"].shape[1])),
                                            "lambda_W":lambda_val,"lambda_H_G":lambda_val,"lambda_H_C":lambda_val,"lambda_Gloss":1,
                                            "G_loss_type":None if 'C-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                            "C_loss_type":None if 'G-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                            "sim":sim,"rank":rank,"init_name":init,"num_init":1,
                                            "G_path":f'{tmp_folder}/G_{variable_name}_{variable}.npy',
                                            "C_path":f'{tmp_folder}/C_{variable_name}_{variable}.npy',
                                            "Z_path":f'{tmp_folder}/Z_{variable_name}_{variable}.npy',"variable_name":variable_name,"variable":variable})
                        else:
                            lambda_val = 0
                            cfgs.append({"run_name":run_name,"G":sim["G"] if not 'C-' in run_name else None,
                                            "C":sim["C"] if not 'G-' in run_name else None,
                                            "Z":sim["Z"] if not 'NMF' in run_name else np.zeros((sim["Z"].shape[0],sim["Z"].shape[1])),
                                            "lambda_W":lambda_val,"lambda_H_G":lambda_val,"lambda_H_C":lambda_val,"lambda_Gloss":1,
                                            "G_loss_type":None if 'C-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                            "C_loss_type":None if 'G-' in run_name else('fro' if 'Fro' in run_name else 'kl_div'),
                                            "sim":sim,"rank":rank,"init_name":init,"num_init":1,
                                            "G_path":f'{tmp_folder}/G_{variable_name}_{variable}.npy',
                                            "C_path":f'{tmp_folder}/C_{variable_name}_{variable}.npy',
                                            "Z_path":f'{tmp_folder}/Z_{variable_name}_{variable}.npy',"variable_name":variable_name,"variable":variable})
        with ProcessPoolExecutor() as pool:
            results_list = list(pool.map(_call_kwargs, cfgs))
        testing_results = pd.concat(results_list, ignore_index=True)
        all_testing_results.append(testing_results)

    all_testing_results = pd.concat(all_testing_results)
    # get summarys dfs
    summary_sim = get_summary_df(all_testing_results,["run_name", "lambda_H_G", "factor_matrix", variable_name,"rank"],"sim")
    # subset to W_C since rel_error recordings for all factor matrices are duplicates (don't differ by factor matrix)
    summary_relerror_G = get_summary_df(all_testing_results[all_testing_results['factor_matrix']=='W_C'],["run_name", "lambda_H_G", variable_name,"rank"],"rel_error_G") 
    summary_relerror_C = get_summary_df(all_testing_results[all_testing_results['factor_matrix']=='W_C'],["run_name", "lambda_H_G", variable_name,"rank"],"rel_error_C")

    summary_sim.to_csv(f'{output_dir}/{variable_name}_summary_sim.csv')
    summary_relerror_G.to_csv(f'{output_dir}/{variable_name}_summary_rel_error_G.csv')
    summary_relerror_C.to_csv(f'{output_dir}/{variable_name}_summary_rel_error_C.csv')
    all_testing_results.to_csv(f'{output_dir}/{variable_name}_all_testing_results.csv')
    if variable_name != 'sparsity':
        tuning_results.to_csv(f'{output_dir}/{variable_name}_tuning_results.csv')
        return summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results
    return summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results

if __name__ == "__main__": 
    # # ASSESS RUNS OVER CORRELATION
    # summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("rho",  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], output_dir=f'{root_dir}/output')

    # # ASSESS RUNS OVER WEIGHT OF COVARIATE SIGNAL
    # summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("ZU_weight",  [0.25,0.5,0.75,1,1.5,2], output_dir=f'{root_dir}/output')

    # # ASSESS RUNS OVER NOISE
    # summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("noise",  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], output_dir=f'{root_dir}/output')

    # ASSESS RUNS OVER SPARSITY
    summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results = run_one("sparsity",  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], output_dir=f'{root_dir}/output')
