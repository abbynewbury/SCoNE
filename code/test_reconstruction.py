#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=test_reconstruction
#SBATCH --nodes=1
#SBATCH --mem=80G
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
import pickle
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
import evaluation.reconstruction_evaluation as reconstruction_evaluation
from simulate_data import *
from utilities import _call_kwargs_deploy_train_run
import pickle

# CODE TO RUN ALL COMPARISON METHODS (AND PARALLELIZE)

def _call_kwargs_run_eval(kw):
    return run_evaluation(**kw)  # expands kwargs dict

def run_evaluation(run_name,file_path,sim, init_name, variable_name, variable, lambda_val, rank):

    with open(f'{file_path}_factor_matrices.pkl', "rb") as f:
        factor_matrices = pickle.load(f)

    with open(f'{file_path}_loss_function.json', "r") as f:
        loss_function = json.load(f)


    results = []
    results_columns = ['run_name','factor_matrix','init','sim']

    if run_name in ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)']:
        G_loss = loss_function['G_loss'][-1]
        C_loss = loss_function['C_loss'][-1]
        if not 'C-' in run_name:
            rel_error_G = G_loss/reconstruction_evaluation.kl_rel_error_denom(sim["G"])
        else: rel_error_G = None
        if not 'G-' in run_name:
            rel_error_C = C_loss/reconstruction_evaluation.kl_rel_error_denom(sim["C"])
        else: rel_error_C = None

    elif run_name == 'MVBC':
        if factor_matrices is False: # run failed for reasons specified in MVBCWrapper
            return pd.DataFrame() 
        rel_error_G = None
        rel_error_C = None
    elif run_name == 'RGWAS':
        if factor_matrices is False: # run failed for reasons specified in RGWASWrapper
            return pd.DataFrame() 
        rel_error_G = None
        rel_error_C = None
    else: assert True == False, f"invalid run name {run_name}"
    for k,v in factor_matrices.items():
        if k=="W":
            results.append([run_name,"W_C",init_name,reconstruction_evaluation.best_permutation_similarity(sim["W_C"],v)])
            results.append([run_name,"W_G",init_name,reconstruction_evaluation.best_permutation_similarity(sim["W_G"],v)])
        else:
            results.append([run_name,k,init_name,reconstruction_evaluation.best_permutation_similarity(sim[k],v)])
    results = pd.DataFrame(results,columns=results_columns)
    results[variable_name] = variable
    results['rel_error_G'] = rel_error_G
    results['rel_error_C'] = rel_error_C
    results['lambda_val'] = lambda_val
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

def is_sparse(run_name): return run_name in {'SCoNE','SCoNE(Fro)','MVBC'} # runs that have sparsity params


def run_one(variable_name, variable_range, output_dir):
    all_results = []
    # ---- simulate data ---- (goes in tmp folder)
    for variable in variable_range:
        sim_kwargs = {"n":500,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":0} 
        sim_kwargs[variable_name] = variable
        sim = simulate_views(**sim_kwargs) 
        with open(f'{tmp_folder}/sim_{variable_name}_{variable}.pkl','wb') as f:
            pickle.dump(sim,f)
        # write G,C,Z to paths
        np.save(f'{tmp_folder}/G_{variable_name}_{variable}',sim["G"])
        np.save(f'{tmp_folder}/C_{variable_name}_{variable}',sim["C"])
        np.save(f'{tmp_folder}/Z_{variable_name}_{variable}',sim["Z"])

    # ---- config -----
    lambda_options = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    tuning_run_names = ['SCoNE' ,'SCoNE(Fro)','MVBC']
    training_run_names = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','SCoNE','SCoNE(Fro)','RGWAS','MVBC']

    # ---- make compact DF ----
    plan = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            init_name=init_name,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_range
        for rn in tuning_run_names
        for init_name in range(10)
        for split in ['tuning']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    new_rows = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            init_name=init_name,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_range
        for rn in training_run_names
        for init_name in range(10)
        for split in ['training']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    plan = pd.concat([plan, new_rows], ignore_index=True)

    # generate job id and outpath
    plan = plan.reset_index(drop=True)
    plan["job_id"] = plan.groupby(['variable','run_name','lambda_option']).ngroup()
    plan["out_path"] = f"{output_dir}/{variable_name}_"+ plan["split"] + "_" + plan["job_id"].astype(str)
    plan.to_csv(f"{output_dir}/{variable_name}_run_plan.csv", index=False)


    # deploy tuning runs
    tuning_args, tuning_rows = [], []
    for idx, row in plan[plan['split']=='tuning'].iterrows():
        G_path = f'{tmp_folder}/G_{variable_name}_{row.variable}.npy'
        C_path = f'{tmp_folder}/C_{variable_name}_{row.variable}.npy'
        Z_path = f'{tmp_folder}/Z_{variable_name}_{row.variable}.npy'
        G = np.load(G_path)
        C = np.load(C_path)
        Z = np.load(Z_path)
        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,variable=row.variable,init_name=row.init_name,
            lambda_Gloss=1, 
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', rank=3, num_init=1, n_jobs=32) 
        tuning_args.append(a)
        tuning_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=1, 
                                    lambda_Gloss=1))
    tune_record = pd.DataFrame(tuning_rows)
    # ---- run TUNE ----
    computation_times_train = {}
    for tune_arg in tuning_args:
        variable_val = tune_arg.pop("variable")
        init_name = tune_arg.pop("init_name")
        job_id, computation_time = _call_kwargs_deploy_train_run(tune_arg)
        computation_times_train[job_id] = computation_time
        # evaluation
        with open(f'{tmp_folder}/sim_{variable_name}_{variable_val}.pkl','rb') as f:
            sim = pickle.load(f)
        results = run_evaluation(tune_arg["run_name"],tune_arg["out_path"],sim, init_name, variable_name, variable_val, tune_arg["reg_params"]["lambda_H_G"], rank=3)
        all_results.append(results)

    tune_record["computation_time"] = tune_record["job_id"].map(computation_times_train)
    tune_record.to_csv(f"{output_dir}/{variable_name}_run_record_tune.csv", index=False)
    print("done with tuning", flush=True)

    # deploy training runs
    training_args, training_rows = [], []
    for idx, row in plan[plan['split']=='training'].iterrows():
        G_path = f'{tmp_folder}/G_{variable_name}_{row.variable}.npy'
        C_path = f'{tmp_folder}/C_{variable_name}_{row.variable}.npy'
        Z_path = f'{tmp_folder}/Z_{variable_name}_{row.variable}.npy'
        G = np.load(G_path)
        C = np.load(C_path)
        Z = np.load(Z_path)
        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,
            lambda_Gloss=1,
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', rank=3, num_init=1) 
        training_args.append(a)
        training_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=1, 
                                    lambda_Gloss=1))
    train_record = pd.DataFrame(training_rows)
    # ---- run TRAIN ----
    computation_times_train = {}
    for train_arg in training_args:
        variable_val = train_arg.pop("variable")
        init_name = train_arg.pop("init_name")
        job_id, computation_time = _call_kwargs_deploy_train_run(train_arg)
        computation_times_train[job_id] = computation_time
        # evaluation
        with open(f'{tmp_folder}/sim_{variable_name}_{variable_val}.pkl','rb') as f:
            sim = pickle.load(f)
        results = run_evaluation(train_arg["run_name"],train_arg["out_path"],sim, init_name, variable_name, variable_val, train_arg["reg_params"]["lambda_H_G"], rank=3)
        all_results.append(results)
    
    train_record["computation_time"] = train_record["job_id"].map(computation_times_train)
    train_record.to_csv(f"{output_dir}/{variable_name}_run_record_tune.csv", index=False)
    print("done with training", flush=True)

    all_results = pd.concat(all_results)
    all_results.to_csv(f'{output_dir}/{variable_name}_results.csv')



if __name__ == "__main__": 
    # ASSESS RUNS OVER CORRELATION -TODO: REMOVE
    summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("rho",  [1.0], output_dir=f'{root_dir}/output/models')

    # ASSESS RUNS OVER CORRELATION
    summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("rho",  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], output_dir=f'{root_dir}/output')

    # # ASSESS RUNS OVER WEIGHT OF COVARIATE SIGNAL
    summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("ZU_weight",  [0.25,0.5,0.75,1.0,1.5,2.0], output_dir=f'{root_dir}/output')

    # # # ASSESS RUNS OVER NOISE
    # summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results, tuning_results = run_one("noise",  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], output_dir=f'{root_dir}/output')

    # ASSESS RUNS OVER SPARSITY
    summary_sim, summary_relerror_G, summary_relerror_C, all_testing_results = run_one("sparsity",  [0,0.5,0.9], output_dir=f'{root_dir}/output')
