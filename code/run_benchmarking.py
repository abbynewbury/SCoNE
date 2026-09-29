import pandas as pd
import os
import json
import numpy as np
import pickle
from simulate_data import *
from scone_tools.algorithms.SCoNE import SCoNE_parallel
from concurrent.futures import ProcessPoolExecutor


workspace_dir='/home/jupyter/workspace/disease_subtyping'
output_dir = f'{workspace_dir}/simulations_output'

DEFAULT_SIM_KWARGS = {"n":5000,"M_C":100,"M_G":100,"rank":3,"noise":0.5,"gamma": 1,"sparsity":0,"rG":0,
                  "rGC":0,"rZ":0,"signed_cov_effects": False,"subgroup_structure": True, "overdispersion_nu":0} 

def run_one_scaling(new_value_dict):
    sim_kwargs = DEFAULT_SIM_KWARGS | new_value_dict
    sim = simulate_views(**sim_kwargs)

    print(sim_kwargs, flush=True)
    print('running on CPU', flush=True) 
    if 'rank' in new_value_dict.keys():
        rank = new_value_dict['rank']
    else:
        rank=3
    factor_matrices, loss_function, benchmark_info = SCoNE_parallel(
        sim["G"].astype(float), 
        sim["C"].astype(float), 
        sim["Z"].astype(float), 
        rank=rank,
        alpha=0,
        lambda_H_G=0, 
        lambda_H_C=0, 
        lambda_Gloss=1,  
        num_init=10,
        init='random',
        G_loss_type='kl_div', 
        C_loss_type='kl_div', 
    )
    benchmark_info['processor'] = 'cpu'
    for k,v in new_value_dict.items():
        benchmark_info[k] = v

    print('running on GPU', flush=True)
    factor_matrices_gpu, loss_function_gpu, benchmark_info_gpu = SCoNE_parallel(
        sim["G"].astype(float), 
        sim["C"].astype(float), 
        sim["Z"].astype(float), 
        rank=rank,
        alpha=0,
        lambda_H_G=0, 
        lambda_H_C=0, 
        lambda_Gloss=1,  
        num_init=10,
        init='random',
        G_loss_type='kl_div', 
        C_loss_type='kl_div', 
        use_gpu=True
    )
    assert np.isclose(loss_function_gpu['total_loss'][-1], 
                      loss_function['total_loss'][-1], rtol=1e-3, atol=1e-6)
    benchmark_info_gpu['processor'] = 'gpu'
    for k,v in new_value_dict.items():
        benchmark_info_gpu[k] = v

    return [benchmark_info, benchmark_info_gpu]

def run_scaling(variable_ranges):
    benchmark_infos = []
    params = list(variable_ranges)
    conditions = [
        dict(zip(params, values))
        for values in zip(*variable_ranges.values())
    ]
    # One benchmark condition at a time.
    # Each condition gets a fresh process.
    with ProcessPoolExecutor(
        max_workers=1,
        max_tasks_per_child=1
    ) as ex:

        for result in ex.map(run_one_scaling, conditions):
            benchmark_infos.extend(result)
    return pd.DataFrame(benchmark_infos)

if __name__ == "__main__":

    benchmark_infos = run_scaling({'M_G':[100,250,500,1000], 
                                   'M_C':[100,250,500,1000]})
    benchmark_infos.to_csv(f'{output_dir}/benchmarking_features.csv') 
    
    benchmark_infos_n = run_scaling({'n':[1000,5000,10000,20000,50000]})
    benchmark_infos_n.to_csv(f'{output_dir}/benchmarking_n.csv')
    
    benchmark_infos_rank = run_scaling({'rank':[2,3,4,5,10]}) 
    benchmark_infos_rank.to_csv(f'{output_dir}/benchmarking_rank.csv')

