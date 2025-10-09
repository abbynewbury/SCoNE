# python wrapper for MVBC
import subprocess
import json
import numpy as np


def MVBCWrapper(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, r_path):
    result = subprocess.run(
    f'{r_path} algorithms/MVBC.R {G_path} {C_path} {rank} {lambda_W} {lambda_H_G} {lambda_H_C}',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True, executable='/bin/bash')
    r_output = result.stdout.strip()
    warn_output = result.stderr.strip()
    if "mvbc does not converge" in r_output:
        return False, "convergence did not improve"
    parsed = json.loads(r_output)
    factor_matrices = {k: np.array(v) for k,v in parsed["factor_matrices"].items()}
    loss_history = {k: [v] for k, v in parsed["loss_history"].items()} 
    return factor_matrices, loss_history
