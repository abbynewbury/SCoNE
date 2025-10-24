# python wrapper for RGWAS
import simulations.genomes1000_sim as sim_functions
import numpy as np
import subprocess
import json

def RGWASWrapper(iid_index_path, r_path, G_path, C_path, Z_path, rank=3, num_init=10):
    result = subprocess.run(
    f'{r_path} algorithms/RGWAS.R {G_path} {C_path} {iid_index_path} {Z_path} {rank} {num_init}',
    stdout=subprocess.PIPE,
    text=True,
    shell=True, executable='/bin/bash')
    r_output = result.stdout.strip()
    start = r_output.find('{')
    end = r_output.rfind('}')
    json_str = r_output[start:end+1]
    parsed = json.loads(json_str)
    if parsed['ll']=="NA": # all runs failed
        return False, "all runs failed (NA log-likelihood)"
    
    factor_matrices = {"W":np.array(parsed['pmat'])}
    loss_function = {'ll':[parsed['ll']]}
    return factor_matrices, loss_function