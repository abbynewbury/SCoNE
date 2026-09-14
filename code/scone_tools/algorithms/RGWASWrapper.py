# python wrapper for RGWAS
import numpy as np
import subprocess
import json
import pickle
from pathlib import Path

def RGWASWrapper(r_path, G_path, C_path, Z_path, rank=3, num_init=10, write_all_init=False, write_all_init_path=''):
    # write all init will indicate to write all random initialization results
    if num_init>1 and write_all_init:
        log_likelihoods = [] 
        factor_matrices_list = []
        loss_function_list = []
        benchmark_info_list = []
        W_list = []
        for run in range(num_init):
            result = subprocess.run(
            f'{r_path} {Path(__file__).resolve().parent}/RGWAS.R {G_path} {C_path} {Z_path} {rank} 1',
            stdout=subprocess.PIPE,
            text=True,
            shell=True, executable='/bin/bash')
            r_output = result.stdout.strip()
            start = r_output.find('{')
            end = r_output.rfind('}')
            json_str = r_output[start:end+1]
            parsed = json.loads(json_str)
            benchmark_info = {"wall_time": parsed["wall_time"]}
            parsed = parsed["result"]
            if parsed['ll']=="NA": # all runs failed
                continue
                
            factor_matrices = {"W":np.array(parsed['pmat'])}
            factor_matrices_list.append(factor_matrices)
            loss_function = {'ll':[parsed['ll']]}
            loss_function_list.append(loss_function)
            log_likelihoods.append(parsed['ll'])
            benchmark_info_list.append(benchmark_info)
            # write all inits
            with open(f"{write_all_init_path}_{run}_loss_function.json", "w") as f: 
                json.dump(loss_function, f)
            with open(f"{write_all_init_path}_{run}_benchmark_info.json", "w") as f: 
                json.dump(benchmark_info, f)
            with open(f"{write_all_init_path}_{run}_factor_matrices.pkl", "wb") as f:
                pickle.dump(factor_matrices, f)

            W_list.append(factor_matrices["W"])
        if len(W_list)==0:
            return False, "all runs failed (NA log-likelihood)"
        # get best manually
        best_run = np.argmax(log_likelihoods)
        factor_matrices = factor_matrices_list[best_run]
        loss_function = loss_function_list[best_run]
        benchmark_info = benchmark_info_list[best_run]

    else:
        result = subprocess.run(
        f'{r_path} {Path(__file__).resolve().parent}/RGWAS.R {G_path} {C_path} {Z_path} {rank} {num_init}',
        stdout=subprocess.PIPE,
        text=True,
        shell=True, executable='/bin/bash')
        r_output = result.stdout.strip()
        start = r_output.find('{')
        end = r_output.rfind('}')
        json_str = r_output[start:end+1]
        parsed = json.loads(json_str)
        benchmark_info = {"wall_time": parsed["wall_time"]}
        parsed = parsed["result"]
        if parsed['ll']=="NA": # all runs failed
            return False, "all runs failed (NA log-likelihood)"
        
        factor_matrices = {"W":np.array(parsed['pmat'])}
        loss_function = {'ll':[parsed['ll']]}

    return factor_matrices, loss_function, benchmark_info