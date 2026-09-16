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
from itertools import product

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

def run_evaluation(run_name, sim, factor_matrices, loss_function):
    '''
    return dictionary with relevant results
    '''
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
def _one_run(tune_arg: dict, tmp_folder: str, split): # this might not actually need variable val
    # Copy so we don't mutate shared state
    ta = dict(tune_arg) 

    sim_id = ta.pop("sim_id")

    job_id = _call_kwargs_deploy_train_run(ta)

    # Load sim (each worker reads its own file)
    sim_path = os.path.join(tmp_folder, f"sim_{split}_{sim_id}.pkl")
    with open(sim_path, "rb") as f:
        sim = pickle.load(f)

    # Load results
    for run in range(ta["num_init"]):
        with open(f'{ta["out_path"]}_{run}_factor_matrices.pkl', "rb") as f:
            factor_matrices = pickle.load(f)
        with open(f'{ta["out_path"]}_{run}_loss_function.json', "r") as f:
            loss_function = json.load(f)

    results = run_evaluation(
        run_name=ta["run_name"],
        file_path=ta["out_path"],
        sim=sim,
    )
    
    results = run_evaluation(
        ta["run_name"],
        ta["out_path"],
        sim,
        ta["num_init"],
        experiment_name,
        ta["reg_params"]["lambda_H_G"],
        rank=3,
        split=split
    )

    return job_id, computation_time, results


def run_one(variable_ranges, output_dir, experiment_name):
    '''
    simulate data, deploy train and tune runs in parallel, evaluate results
    try every combination of variables and their corresponding ranges in variable_ranges
    '''
    param_names = list(variable_ranges)
    
    conditions = pd.DataFrame(
        product(*variable_ranges.values()),
        columns=param_names,
    )
    conditions["sim_id"] = range(len(conditions))
    
    all_results = []
    DEFAULT_SIM_KWARGS = {"n":1200,"M_C":20,"num_genes":20,"rank":3,"noise":0.5,"gamma": 5,"sparsity":0.2,"rG":0.5,
                      "rGC":0.2,"rZ":0.2,"signed_cov_effects": False,"subgroup_structure": True, "overdispersion_nu":0} 
    
    # ---- simulate data ---- (goes in tmp folder)
    for _, condition in conditions.iterrows():

        sim_kwargs = DEFAULT_SIM_KWARGS | {
        k: condition[k] for k in param_names}

        for split, seed in [("tune", 0), ("train", 1)]:
            sim_kwargs["seed"] = seed
            sim = simulate_views(**sim_kwargs)

            sim_id = condition["sim_id"]

            with open(f'{tmp_folder}/sim_{split}_{sim_id}.pkl','wb') as f:
                pickle.dump(sim,f)
            
            # write G,C,Z to paths
            pd.DataFrame(sim["G"]).to_csv(f'{tmp_folder}/G_{split}_{sim_id}.csv')
            pd.DataFrame(sim["C"]).to_csv(f'{tmp_folder}/C_{split}_{sim_id}.csv')
            pd.DataFrame(sim["Z"]).to_csv(f'{tmp_folder}/Z_{split}_{sim_id}.csv')


    # ---- config -----
    lambda_options = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    tuning_run_names = ['SCoNE','MVBC']
    training_run_names = ['SCoNE','MVBC','HNMF','HNMF(res)','RGWAS'] 

    # ---- make compact DF ----
    plan = pd.DataFrame(
        [dict(sim_id=sim_id,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for sim_id in conditions['sim_id']
        for rn in tuning_run_names
        for split in ['tune']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    new_rows = pd.DataFrame(
        [dict(sim_id=sim_id,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for sim_id in conditions['sim_id']
        for rn in training_run_names
        for split in ['train']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    plan = pd.concat([plan, new_rows], ignore_index=True)

    # generate job id and outpath
    plan = plan.merge(conditions, on="sim_id", how="left")
    plan = plan.reset_index(drop=True)  
    plan["job_id"] = plan.groupby(['sim_id','run_name','lambda_option']).ngroup()
    plan["out_path"] = f"{output_dir}/models/{experiment_name}_"+ plan["split"] + "_" + plan["job_id"].astype(str)
    plan.to_csv(f"{output_dir}/models/{experiment_name}_run_plan.csv", index=False)


    # deploy tuning runs
    tuning_args, tuning_rows = [], []
    for idx, row in plan[plan['split']=='tune'].iterrows():
        G_path = f'{tmp_folder}/G_tune_{row.sim_id}.csv'
        C_path = f'{tmp_folder}/C_tune_{row.sim_id}.csv'
        Z_path = f'{tmp_folder}/Z_tune_{row.sim_id}.csv' # TODO: pick up here with new sim id plan
        G = pd.read_csv(G_path, index_col=0).to_numpy(dtype=np.float64)
        C = pd.read_csv(C_path, index_col=0).to_numpy(dtype=np.float64)
        Z = pd.read_csv(Z_path, index_col=0).to_numpy(dtype=np.float64)

        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,sim_id=row.sim_id,
            lambda_Gloss=1, # TODO: rscript package download
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/gpfs/commons/home/anewbury/miniconda/bin/Rscript', rank=3, num_init=50,
            write_all_init=True, write_all_init_path=row.out_path) 
        tuning_args.append(a)
        tuning_rows.append(dict(out_path=row.out_path,sim_id=row.sim_id,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=50, 
                                    lambda_Gloss=1))
    tune_record = pd.DataFrame(tuning_rows)
    # ---- run TUNE ----
    computation_times_tune = {}

    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(_one_run, tune_arg, tmp_folder, 'tune')
            for tune_arg in tuning_args
        ]

        for fut in as_completed(futures):
            job_id, computation_time, results = fut.result()
            computation_times_tune[job_id] = computation_time
            results['job_id'] = job_id
            all_results.append(results)

    #tune_record["computation_time"] = tune_record["job_id"].map(computation_times_tune) - ADD THIS TO RESULTS
    tune_record.to_csv(f"{output_dir}/models/{variable_name}_run_record_tune.csv", index=False)
    print("done with tuning", flush=True)
    # TODO GET BEST PERFORMING LAMBDA

    # deploy training runs
    training_args, training_rows = [], []
    for idx, row in plan[plan['split']=='train'].iterrows():
        G_path = f'{tmp_folder}/G_train_{row.sim_id}.csv'
        C_path = f'{tmp_folder}/C_train_{row.sim_id}.csv'
        Z_path = f'{tmp_folder}/Z_train_{row.sim_id}.csv'
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
        # ASSESS RUNS OVER rG 
        run_one({"rG":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, output_dir=output_dir, experiment_name="rG")

        # ASSESS RUNS OVER rGC
        run_one({"rGC":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, output_dir=output_dir, experiment_name="rGC")

        # ASSESS RUNS OVER rZ
        run_one({"rZ":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, output_dir=output_dir, experiment_name="rZ_unobs") # TODO: add w and w out obs conf

        # ASSESS RUNS OVER SIGNED EFFECTS
        run_one({"signed_cov_effects":[True, False]}, output_dir=output_dir, experiment_name="signed_cov_effects")

        # ASSESS RUNS OVER OVERDISPERSION
        run_one({"overdispersion_nu":[0, 0.1, 0.5, 1.0]}, output_dir=output_dir, experiment_name="overdispersion_nu")

        # ASSESS RUNS OVER RANK MISSPECIFICATION
        run_one({"rank":[2, 3, 4, 5], "subgroup_structure":[True, False]}, output_dir=output_dir, experiment_name="rank_subgroupstructure") 
        
        # Supplementary

        # ASSESS RUNS OVER SPARSITY
        run_one({"sparsity": [0.0,0.25,0.5,0.75,0.9]}, output_dir=output_dir, experiment_name="sparsity")

        # ASSESS RUNS OVER GAMMA
        run_one({"gamma": [1,5,10]}, output_dir=output_dir,  experiment_name="gamma")
        
    # else:
    #     variable_names = ["rho", "ZU_weight", "noise", "sparsity"]
    #     for variable_name in variable_names:
    #         print(variable_name)
    #         all_results = []
    #         # get tune results
    #         split = "tune"
    #         tune_record = pd.read_csv(f"{output_dir}/models/{variable_name}_run_record_tune.csv")
    #         tune_record['out_path'] = tune_record['out_path'].str.replace('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno',root_dir)
    #         for i, row in tune_record.iterrows():
    #             variable_val = row["variable"]
    #             sim_path = os.path.join(tmp_folder, f"sim_{split}_{variable_name}_{variable_val}.pkl")
    #             with open(sim_path, "rb") as f:
    #                 sim = pickle.load(f)
    #             results = run_evaluation(
    #                 row["run_name"],
    #                 row["out_path"],
    #                 sim,
    #                 row["num_init"],
    #                 variable_name,
    #                 variable_val,
    #                 row["lambda_option"],
    #                 rank=3,
    #                 split=split
    #             )
    #             results['job_id'] = row['job_id']
    #             all_results.append(results)

    #         # get train results
    #         split = "train"
    #         train_record = pd.read_csv(f"{output_dir}/models/{variable_name}_run_record_train.csv")
    #         train_record['out_path'] = train_record['out_path'].str.replace('/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno',root_dir)
    #         for i, row in train_record.iterrows():
    #             variable_val = row["variable"]
    #             sim_path = os.path.join(tmp_folder, f"sim_{split}_{variable_name}_{variable_val}.pkl")
    #             with open(sim_path, "rb") as f:
    #                 sim = pickle.load(f)
    #             results = run_evaluation(
    #                 row["run_name"],
    #                 row["out_path"],
    #                 sim,
    #                 row["num_init"],
    #                 variable_name,
    #                 variable_val,
    #                 row["lambda_option"],
    #                 rank=3,
    #                 split=split
    #             )
    #             results['job_id'] = row['job_id']
    #             all_results.append(results)
    #         all_results = pd.concat(all_results)
    #         all_results.to_csv(f'{output_dir}/{variable_name}_results.csv')
    