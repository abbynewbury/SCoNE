import pandas as pd
from itertools import product
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment
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
import scone_tools.evaluation.reconstruction_evaluation as reconstruction_evaluation
import scone_tools.evaluation.cluster_evaluation as cluster_evaluation
from simulate_data import *
from scone_tools.utilities import _call_kwargs_deploy_train_run
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


# CODE TO RUN ALL COMPARISON METHODS (AND PARALLELIZE)
'''
This code not meant to be run on RGWAS because the simulations are designed to evaluate recovery under a shared low-rank factor structure and therefore do not provide a directly comparable evaluation setting for RGWAS, which is not a matrix factorization method
'''

def _call_kwargs_run_eval(kw):
    return run_evaluation(**kw)  # expands kwargs dict

def run_evaluation(run_name, sim, factor_matrices, loss_function):
    '''
    return dictionary with relevant results
    '''
    results = {}

    if factor_matrices is False: # run failed for reasons specified in MVBCWrapper
        return {}

    # calculate subgroup detection
    if "W" in sim.keys():
        ari = cluster_evaluation.adjusted_rand_index(sim["W"],factor_matrices["W"])
        results['ari'] = ari
        _, optimal_permutation = reconstruction_evaluation.best_permutation_similarity(sim["W"],factor_matrices["W"]) # calculate optimal permutation on W and keep for rest
    else: ari=None

    # calculate reconstruction
    for k,v in factor_matrices.items():
        if "W" in sim.keys():
            if k in ['W','H_G','H_C']:
                results[f'sim_{k}'] = reconstruction_evaluation.frobenius_cosine_similarity(sim[k],v[:,optimal_permutation])
                results[f'sim_{k}_null'] = np.mean([reconstruction_evaluation.frobenius_cosine_similarity(sim[k],
                                                    v[np.random.permutation(len(v)),:][:, optimal_permutation]) 
                                                    for _ in range(100)])
        if k in ['U_G','U_C']:
            results[f'sim_{k}'] = reconstruction_evaluation.frobenius_cosine_similarity(sim[k],v)
            results[f'sim_{k}_null'] = np.mean([reconstruction_evaluation.frobenius_cosine_similarity(sim[k],
                                                v[np.random.permutation(len(v)),:]) 
                                                for _ in range(100)])
    
    # get tuning metrics 
    if run_name !='MVBC':
        G_loss = loss_function['G_loss'][-1]
        C_loss = loss_function['C_loss'][-1]
        if run_name != 'HNMF(res)':
            results['rel_error_G'] = G_loss/reconstruction_evaluation.kl_rel_error_denom(sim["G"])
            results['rel_error_C'] = C_loss/reconstruction_evaluation.kl_rel_error_denom(sim["C"])
        else:
            results['rel_error_G'] = G_loss/np.linalg.norm(sim["G"])**2
            results['rel_error_C'] = C_loss/np.linalg.norm(sim["C"])**2
        results['tuning_metric'] = results['rel_error_G']+results['rel_error_C'] # only ones with sparsity params need tuning metric
    else:
        # with no subgroup structure, only looking at CCC anyway which can't be calculated for MVBC, so MVBC won't be run in this case
        results['tuning_metric'] = -results['ari'] # search for min value

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
def _one_run(tune_arg: dict, sim_folder: str, split: str): 
    # Copy so we don't mutate shared state
    ta = dict(tune_arg) 
    
    sim_id = ta.pop("sim_id")
    
    job_id = _call_kwargs_deploy_train_run(ta)
    
    # Load sim (each worker reads its own file)
    sim_path = os.path.join(sim_folder, f"sim_{split}_{sim_id}.pkl")
    with open(sim_path, "rb") as f:
        sim = pickle.load(f)
    
    assert np.allclose(ta['G'],sim['G']), "G in method not linked with simulation id"
    assert np.allclose(ta['C'],sim['C']), "C in method not linked with simulation id"
    assert np.allclose(ta['Z'],sim['Z']), "Z in method not linked with simulation id"
    
    # Load results
    all_run_results = []
    runs = range(ta["num_init"]) if ta["run_name"] != "MVBC" else [0]
    for run in runs:
        suffix = f"_{run}" if ta["run_name"] != 'MVBC' else ""
        with open(f'{ta["out_path"]}{suffix}_factor_matrices.pkl', "rb") as f:
            factor_matrices = pickle.load(f)
        with open(f'{ta["out_path"]}{suffix}_loss_function.json', "r") as f:
            loss_function = json.load(f)
    
        results = run_evaluation(ta["run_name"], sim, factor_matrices, loss_function)
        results['run'] = run
        results['split'] = split
        results['sim_id'] = sim_id
        results['job_id'] = job_id
        all_run_results.append(results)

    all_run_results = pd.DataFrame(all_run_results)
    if ta["run_name"] != 'MVBC':
        with open(f'{ta["out_path"]}_loss_function.json', "r") as f:
            loss_function = json.load(f)
        all_run_results['coph_corr'] = loss_function['coph_corr'] # over all runs
    
    return all_run_results


def run_one(variable_ranges, output_dir, experiment_name):
    '''
    simulate data, deploy train and tune runs in parallel, evaluate results
    try every combination of variables and their corresponding ranges in variable_ranges
    '''
    exp_tmp_folder = f'{tmp_folder}/{experiment_name}'
    exp_models_folder = f'{output_dir}/models/{experiment_name}'
    os.makedirs(exp_tmp_folder, exist_ok=True)
    os.makedirs(exp_models_folder, exist_ok=True)

    param_names = list(variable_ranges)
    
    conditions = pd.DataFrame(
        product(*variable_ranges.values()),
        columns=param_names,
    )
    conditions["sim_id"] = range(len(conditions))
    conditions["sim_id"] = conditions["sim_id"].astype(int)
    print(conditions, flush=True)
    
    all_results = []
    # TODO: put back once testing done
    # DEFAULT_SIM_KWARGS = {"n":1200,"M_C":20,"M_G":20,"rank":3,"noise":0.5,"gamma": 5,"sparsity":0.2,"rG":0.5,
    #                   "rGC":0.2,"rZ":0.2,"signed_cov_effects": False,"subgroup_structure": True, "overdispersion_nu":0} 
    DEFAULT_SIM_KWARGS = {"n":1200,"M_C":20,"M_G":20,"rank":3,"noise":0.5,"gamma": 1, "sparsity":0,"rG":0,
                      "rGC":0,"rZ":0,"signed_cov_effects": False,"subgroup_structure": True, "overdispersion_nu":0} 
    
    # ---- simulate data ---- (goes in tmp folder)
    for _, condition in conditions.iterrows():

        sim_kwargs = DEFAULT_SIM_KWARGS | {
        k: condition[k] for k in param_names if k!='unobs_conf'} # intentional: w/ or w/out unobs_conf won't change sim even though given different id

        for split, seed in [("tune", 0), ("train", 1)]:
            sim_kwargs["seed"] = seed
            sim = simulate_views(**sim_kwargs)

            sim_id = int(condition["sim_id"])

            if 'unobs_conf' in param_names:
                # record
                pd.DataFrame(sim["Z"]).to_csv(f'{exp_tmp_folder}/Z_{split}_{sim_id}_w_conf.csv')
                # remove covariate
                sim["Z"] = sim['Z'][:, np.all(sim['Z'] == 1, axis=0)]

            with open(f'{exp_tmp_folder}/sim_{split}_{sim_id}.pkl','wb') as f:
                pickle.dump(sim,f)
            
            # write G,C,Z to paths
            pd.DataFrame(sim["G"]).to_csv(f'{exp_tmp_folder}/G_{split}_{sim_id}.csv')
            pd.DataFrame(sim["C"]).to_csv(f'{exp_tmp_folder}/C_{split}_{sim_id}.csv')
            pd.DataFrame(sim["Z"]).to_csv(f'{exp_tmp_folder}/Z_{split}_{sim_id}.csv')


    # ---- config -----
    lambda_options = [0, 1e-3, 1e-2, 1e-1, 1]
    tuning_run_names = ['SCoNE','MVBC']
    training_run_names = ['SCoNE','MVBC','HNMF','HNMF(res)'] 
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
    if 'subgroup_structure' in param_names:
        plan = plan[(plan['run_name']!='MVBC')|(plan['subgroup_structure'])].copy() # only run MVBC with subgroup structure
    if 'unobs_conf' in param_names:
        plan = plan[(~plan['unobs_conf'])|(plan['run_name']=='SCoNE')].copy() # only run SCoNE if including unobs confounding
    plan = plan.reset_index(drop=True)  
    plan["job_id"] = plan.groupby(['sim_id','run_name','lambda_option']).ngroup()
    plan["out_path"] = exp_models_folder + "/"+ plan["split"] + "_" + plan["job_id"].astype(str)
    print(f'plan dtypes: {plan.dtypes}',flush=True)
    plan.to_csv(f"{exp_models_folder}/run_plan.csv", index=False)
    
    
    # deploy tuning runs
    tuning_args, tuning_rows = [], []
    for idx, row in plan[plan['split']=='tune'].iterrows():
        G_path = f'{exp_tmp_folder}/G_tune_{row.sim_id}.csv'
        C_path = f'{exp_tmp_folder}/C_tune_{row.sim_id}.csv'
        Z_path = f'{exp_tmp_folder}/Z_tune_{row.sim_id}.csv' 
        G = pd.read_csv(G_path, index_col=0).to_numpy(dtype=np.float64)
        C = pd.read_csv(C_path, index_col=0).to_numpy(dtype=np.float64)
        Z = pd.read_csv(Z_path, index_col=0).to_numpy(dtype=np.float64)
    
        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,sim_id=row.sim_id,
            lambda_Gloss=1, 
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/opt/conda/envs/jupyter/bin/Rscript', rank=3, num_init=50,
            write_all_init=True, write_all_init_path=row.out_path) 
        tuning_args.append(a)
        tuning_rows.append(dict(out_path=row.out_path,sim_id=row.sim_id,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=50, 
                                    lambda_Gloss=1))
    tune_record = pd.DataFrame(tuning_rows)
    
    # ---- run TUNE ----

    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(_one_run, tune_arg, exp_tmp_folder, 'tune')
            for tune_arg in tuning_args
        ]

        for fut in as_completed(futures):
            results = fut.result()
            all_results.append(results)

    tune_record.to_csv(f"{exp_models_folder}/run_record_tune.csv", index=False)
    print("done with tuning", flush=True)
    tune_results = pd.concat(all_results).merge(tune_record,on=['sim_id','job_id'],how='left')
    optimal_job_ids = tune_results.loc[tune_results.groupby(['sim_id','run_name'])['tuning_metric'].idxmin()]['job_id']

    # deploy training runs
    training_args, training_rows = [], []
    for idx, row in plan[(plan['split']=='train')&(plan['job_id'].isin(optimal_job_ids)|~plan["run_name"].apply(is_sparse))].iterrows():
        G_path = f'{exp_tmp_folder}/G_train_{row.sim_id}.csv'
        C_path = f'{exp_tmp_folder}/C_train_{row.sim_id}.csv'
        Z_path = f'{exp_tmp_folder}/Z_train_{row.sim_id}.csv'
        G = pd.read_csv(G_path, index_col=0).to_numpy(dtype=np.float64)
        C = pd.read_csv(C_path, index_col=0).to_numpy(dtype=np.float64)
        Z = pd.read_csv(Z_path, index_col=0).to_numpy(dtype=np.float64)

        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,sim_id=row.sim_id,
            lambda_Gloss=1,
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path='/opt/conda/envs/jupyter/bin/Rscript', rank=3, num_init=50, 
            write_all_init=True, write_all_init_path=row.out_path) 
        training_args.append(a)
        training_rows.append(dict(out_path=row.out_path,sim_id=row.sim_id,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=50,
                                    lambda_Gloss=1))
    train_record = pd.DataFrame(training_rows)
    # ---- run TRAIN ----
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [
            ex.submit(_one_run, train_arg, exp_tmp_folder, 'train')
            for train_arg in training_args
        ]

        for fut in as_completed(futures):
            results = fut.result()
            all_results.append(results)
    
    train_record.to_csv(f"{exp_models_folder}/run_record_train.csv", index=False)
    print("done with training", flush=True)

    all_results = pd.concat(all_results)
    all_results = all_results.merge(plan, how='left', on=['split','sim_id','job_id'])
    all_results.to_csv(f'{output_dir}/{experiment_name}_results.csv') 



if __name__ == "__main__": 
    run_evaluation_only=False
    output_dir = f'{root_dir}/output'

    # ASSESS RUNS OVER rG 
    run_one({"rG":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, output_dir=output_dir, experiment_name="rG")

    # ASSESS RUNS OVER rGC
    run_one({"rGC":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, output_dir=output_dir, experiment_name="rGC")

    # ASSESS RUNS OVER rZ
    run_one({"rZ":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], "unobs_conf":[True,False]}, output_dir=output_dir, experiment_name="rZ_unobs") 

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
        

    