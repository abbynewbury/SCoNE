# python wrapper for MVBC
import subprocess
import json


def MVBCWrapper(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, r_path):
    result = subprocess.run(
    f'{r_path} algorithms/MVBC.R {G_path} {C_path} {rank} {lambda_W} {lambda_H_G} {lambda_H_C}',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True, executable='/bin/bash')
    r_output = result.stdout.strip()
    warn_output = result.stderr.strip()
    assert "converg" not in warn_output, "Issue with convergence"   
    parsed = json.loads(r_output)
    factor_matrices = parsed["factor_matrices"]
    loss_history = parsed["loss_history"]
    return factor_matrices, loss_history
