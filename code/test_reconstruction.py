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

# make tmp dir
root_dir = '/home/jupyter/repos/SCoNE'
code_dir = f'{root_dir}/code'
tmp_folder = f"{root_dir}/output/tmp"
os.makedirs(tmp_folder, exist_ok=True)
sys.path.append(code_dir)
import evaluation.reconstruction_evaluation as reconstruction_evaluation
import evaluation.cluster_evaluation as cluster_evaluation
from simulate_data import *
from utilities import _call_kwargs_deploy_train_run
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


# CODE TO RUN ALL COMPARISON METHODS (AND PARALLELIZE)

def _call_kwargs_run_eval(kw):
    return run_evaluation(**kw)  # expands kwargs dict

def run_evaluation(run_name,file_path,sim, num_init, variable_name, variable, lambda_val, rank, split):
    results = []
    results_columns = ['run_name','factor_matrix','init','sim','rel_error_G','rel_error_C']
    if run_name == 'MVBC':
        with open(f'{file_path}_factor_matrices.pkl', "rb") as f:
            factor_matrices = pickle.load(f)

        try:
            with open(f'{file_path}_loss_function.json', "r") as f:
                loss_function = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSON decode failed for: {file_path}_loss_function.json\n"
            ) from e

        if factor_matrices is False: # run failed for reasons specified in MVBCWrapper
            return pd.DataFrame() 
        rel_error_G = None
        rel_error_C = None
        for k,v in factor_matrices.items():
            if k=="W":
                # get optimal permutation
                _, optimal_permutation = reconstruction_evaluation.best_permutation_similarity(sim["W_C"],v)
                results.append([run_name,"W_C",0,reconstruction_evaluation.frobenius_cosine_similarity(sim["W_C"],v[:,optimal_permutation]),rel_error_G,rel_error_C])
                results.append([run_name,"W_G",0,reconstruction_evaluation.frobenius_cosine_similarity(sim["W_G"],v[:,optimal_permutation]),rel_error_G,rel_error_C])
            else:
                assert True==False, "MVBC returning something other than W"
    else:
        for run in range(num_init):
            with open(f'{file_path}_{run}_factor_matrices.pkl', "rb") as f:
                factor_matrices = pickle.load(f)

            try:
                with open(f'{file_path}_{run}_loss_function.json', "r") as f:
                    loss_function = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"JSON decode failed for: {file_path}_loss_function.json\n"
                ) from e


            if run_name in ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)']:
                G_loss = loss_function['G_loss'][-1]
                C_loss = loss_function['C_loss'][-1]
                if not 'C-' in run_name:
                    rel_error_G = G_loss/reconstruction_evaluation.kl_rel_error_denom(sim["G"])
                else: rel_error_G = None
                if not 'G-' in run_name:
                    rel_error_C = C_loss/reconstruction_evaluation.kl_rel_error_denom(sim["C"])
                else: rel_error_C = None

            elif run_name == 'RGWAS':
                if factor_matrices is False: # run failed for reasons specified in RGWASWrapper
                    return pd.DataFrame() 
                rel_error_G = None
                rel_error_C = None
            else: assert True == False, f"invalid run name {run_name}"
            _, optimal_permutation = reconstruction_evaluation.best_permutation_similarity(sim["W_C"],factor_matrices["W"]) # calculate optimal permutation once on W and keep for rest
            for k,v in factor_matrices.items():
                if k=="W":
                    results.append([run_name,"W_C",run,reconstruction_evaluation.frobenius_cosine_similarity(sim["W_C"],v[:,optimal_permutation]),rel_error_G,rel_error_C])
                    results.append([run_name,"W_G",run,reconstruction_evaluation.frobenius_cosine_similarity(sim["W_G"],v[:,optimal_permutation]),rel_error_G,rel_error_C])
                elif k in ['H_G', 'H_C']:
                    results.append([run_name,k,run,reconstruction_evaluation.frobenius_cosine_similarity(sim[k],v[:,optimal_permutation]),rel_error_G,rel_error_C])
                else:
                    results.append([run_name,k,run,reconstruction_evaluation.frobenius_cosine_similarity(sim[k],v),rel_error_G,rel_error_C])
    # calculate CCC
    results = pd.DataFrame(results,columns=results_columns)
    results[variable_name] = variable
    results['lambda_val'] = lambda_val
    results['split'] = split
    results['rank'] = rank
    return results

def get_summary_df(df,grouping_vars,value_var):
    summary = (
        df.groupby(grouping_vars)[value_var]
            .agg(n='count', mean='mean', std='std').reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary["ymin"] = summary["mean"] - summary["se"]
    summary["ymax"] = summary["mean"] + summary["se"]
    return summary

def is_sparse(run_name): return run_name in {'SCoNE','SCoNE(Fro)','MVBC'} # runs that have sparsity params


# to run in parallel (one run here signifies one run within a specified variable name and range)
def _one_run(tune_arg: dict, tmp_folder: str, variable_name: str, split):
    # Copy so we don't mutate shared state
    ta = dict(tune_arg)

    variable_val = ta.pop("variable")

    job_id, computation_time = _call_kwargs_deploy_train_run(ta)

    # Load sim (each worker reads its own file)
    sim_path = os.path.join(tmp_folder, f"sim_{split}_{variable_name}_{variable_val}.pkl")
    with open(sim_path, "rb") as f:
        sim = pickle.load(f)

    results = run_evaluation(
        ta["run_name"],
        ta["out_path"],
        sim,
        ta["num_init"],
        variable_name,
        variable_val,
        ta["reg_params"]["lambda_H_G"],
        rank=3,
        split=split
    )

    return job_id, computation_time, results



def run_one(variable_name, variable_range, output_dir):
    all_results = []
    # ---- simulate data ---- (goes in tmp folder)
    for variable in variable_range:
        # tuning
        sim_kwargs = {"n":1000,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":0} 
        sim_kwargs[variable_name] = variable
        sim = simulate_views(**sim_kwargs) 
        with open(f'{tmp_folder}/sim_tune_{variable_name}_{variable}.pkl','wb') as f:
            pickle.dump(sim,f)
        # write G,C,Z to paths
        pd.DataFrame(sim["G"]).to_csv(f'{tmp_folder}/G_tune_{variable_name}_{variable}.csv')
        pd.DataFrame(sim["C"]).to_csv(f'{tmp_folder}/C_tune_{variable_name}_{variable}.csv')
        pd.DataFrame(sim["Z"]).to_csv(f'{tmp_folder}/Z_tune_{variable_name}_{variable}.csv')

        # training
        sim_kwargs = {"n":500,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":1} 
        sim_kwargs[variable_name] = variable
        sim = simulate_views(**sim_kwargs) 
        with open(f'{tmp_folder}/sim_train_{variable_name}_{variable}.pkl','wb') as f:
            pickle.dump(sim,f)
        # write G,C,Z to paths
        pd.DataFrame(sim["G"]).to_csv(f'{tmp_folder}/G_train_{variable_name}_{variable}.csv')
        pd.DataFrame(sim["C"]).to_csv(f'{tmp_folder}/C_train_{variable_name}_{variable}.csv')
        pd.DataFrame(sim["Z"]).to_csv(f'{tmp_folder}/Z_train_{variable_name}_{variable}.csv')

    # ---- config -----
    lambda_options = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    tuning_run_names = ['SCoNE'] #['SCoNE' ,'SCoNE(Fro)','MVBC']
    training_run_names = ['SCoNE'] # ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','SCoNE','SCoNE(Fro)','RGWAS','MVBC']

    # ---- make compact DF ----
    plan = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_range
        for rn in tuning_run_names
        for split in ['tune']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    new_rows = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_range
        for rn in training_run_names
        for split in ['train']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    plan = pd.concat([plan, new_rows], ignore_index=True)

    # generate job id and outpath
    plan = plan.reset_index(drop=True)
    plan["job_id"] = plan.groupby(['variable','run_name','lambda_option']).ngroup()
    plan["out_path"] = f"{output_dir}/models/{variable_name}_"+ plan["split"] + "_" + plan["job_id"].astype(str)
    plan.to_csv(f"{output_dir}/models/{variable_name}_run_plan.csv", index=False)


    # deploy tuning runs
    tuning_args, tuning_rows = [], []
    for idx, row in plan[plan['split']=='tune'].iterrows():
        G_path = f'{tmp_folder}/G_tune_{variable_name}_{row.variable}.csv'
        C_path = f'{tmp_folder}/C_tune_{variable_name}_{row.variable}.csv'
        Z_path = f'{tmp_folder}/Z_tune_{variable_name}_{row.variable}.csv'
        G = pd.read_csv(G_path, index_col=0).to_numpy(dtype=np.float64)
        C = pd.read_csv(C_path, index_col=0).to_numpy(dtype=np.float64)
        Z = pd.read_csv(Z_path, index_col=0).to_numpy(dtype=np.float64)

        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,variable=row.variable,
            lambda_Gloss=1, 
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', rank=3, num_init=50,
            write_all_init=True, write_all_init_path=row.out_path) 
        tuning_args.append(a)
        tuning_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=50, 
                                    lambda_Gloss=1))
    tune_record = pd.DataFrame(tuning_rows)
    # ---- run TUNE ----
    computation_times_tune = {}

    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(_one_run, tune_arg, tmp_folder, variable_name, 'tune')
            for tune_arg in tuning_args
        ]

        for fut in as_completed(futures):
            job_id, computation_time, results = fut.result()
            computation_times_tune[job_id] = computation_time
            results['job_id'] = job_id
            all_results.append(results)

    tune_record["computation_time"] = tune_record["job_id"].map(computation_times_tune)
    tune_record.to_csv(f"{output_dir}/models/{variable_name}_run_record_tune.csv", index=False)
    print("done with tuning", flush=True)

    # deploy training runs
    training_args, training_rows = [], []
    for idx, row in plan[plan['split']=='train'].iterrows():
        G_path = f'{tmp_folder}/G_train_{variable_name}_{row.variable}.csv'
        C_path = f'{tmp_folder}/C_train_{variable_name}_{row.variable}.csv'
        Z_path = f'{tmp_folder}/Z_train_{variable_name}_{row.variable}.csv'
        G = pd.read_csv(G_path, index_col=0).to_numpy(dtype=np.float64)
        C = pd.read_csv(C_path, index_col=0).to_numpy(dtype=np.float64)
        Z = pd.read_csv(Z_path, index_col=0).to_numpy(dtype=np.float64)

        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,variable=row.variable,
            lambda_Gloss=1,
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', rank=3, num_init=50,
            write_all_init=True, write_all_init_path=row.out_path) 
        training_args.append(a)
        training_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=50, 
                                    lambda_Gloss=1))
    train_record = pd.DataFrame(training_rows)
    # ---- run TRAIN ----
    computation_times_train = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(_one_run, train_arg, tmp_folder, variable_name, 'train')
            for train_arg in training_args
        ]

        for fut in as_completed(futures):
            job_id, computation_time, results = fut.result()
            computation_times_train[job_id] = computation_time
            results['job_id'] = job_id
            all_results.append(results)
    
    train_record["computation_time"] = train_record["job_id"].map(computation_times_train)
    train_record.to_csv(f"{output_dir}/models/{variable_name}_run_record_train.csv", index=False)
    print("done with training", flush=True)

    all_results = pd.concat(all_results)
    all_results.to_csv(f'{output_dir}/{variable_name}_results.csv')



if __name__ == "__main__": 
    run_evaluation_only=False
    output_dir = f'{root_dir}/output'

    if not run_evaluation_only:
        # ASSESS RUNS OVER CORRELATION 
        run_one("rho",  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], output_dir=output_dir)

        # ASSESS RUNS OVER WEIGHT OF COVARIATE SIGNAL
        run_one("ZU_weight",  [0.25,0.5,0.75,1.0,1.5,2.0], output_dir=output_dir)

        # ASSESS RUNS OVER NOISE
        run_one("noise",  [0.0,0.5,1.0], output_dir=output_dir)

        # ASSESS RUNS OVER SPARSITY
        run_one("sparsity",  [0.0,0.25,0.5,0.75,0.9], output_dir=output_dir)

    else:
        variable_names = ["rho", "ZU_weight", "noise", "sparsity"]
        for variable_name in variable_names:
            print(variable_name)
            all_results = []
            # get tune results
            split = "tune"
            tune_record = pd.read_csv(f"{output_dir}/models/{variable_name}_run_record_tune.csv")
            tune_record['out_path'] = tune_record['out_path'].str.replace('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno',root_dir)
            for i, row in tune_record.iterrows():
                variable_val = row["variable"]
                sim_path = os.path.join(tmp_folder, f"sim_{split}_{variable_name}_{variable_val}.pkl")
                with open(sim_path, "rb") as f:
                    sim = pickle.load(f)
                results = run_evaluation(
                    row["run_name"],
                    row["out_path"],
                    sim,
                    row["num_init"],
                    variable_name,
                    variable_val,
                    row["lambda_option"],
                    rank=3,
                    split=split
                )
                results['job_id'] = row['job_id']
                all_results.append(results)

            # get train results
            split = "train"
            train_record = pd.read_csv(f"{output_dir}/models/{variable_name}_run_record_train.csv")
            train_record['out_path'] = train_record['out_path'].str.replace('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno',root_dir)
            for i, row in train_record.iterrows():
                variable_val = row["variable"]
                sim_path = os.path.join(tmp_folder, f"sim_{split}_{variable_name}_{variable_val}.pkl")
                with open(sim_path, "rb") as f:
                    sim = pickle.load(f)
                results = run_evaluation(
                    row["run_name"],
                    row["out_path"],
                    sim,
                    row["num_init"],
                    variable_name,
                    variable_val,
                    row["lambda_option"],
                    rank=3,
                    split=split
                )
                results['job_id'] = row['job_id']
                all_results.append(results)
            all_results = pd.concat(all_results)
            all_results.to_csv(f'{output_dir}/{variable_name}_results.csv')
    