# python wrapper for MVBC
import subprocess
import json
import numpy as np
from pathlib import Path


def MVBCWrapper(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, r_path):
    result = subprocess.run(
    f'{r_path} {Path(__file__).resolve().parent}/MVBC.R {G_path} {C_path} {rank} {lambda_W} {lambda_H_G} {lambda_H_C}',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    shell=True, executable='/bin/bash')

    r_output = result.stdout.strip()
    warn_output = result.stderr.strip()
    if "mvbc does not converge" in r_output:
        return False, "mvbc does not converge", None
    json_str = r_output[r_output.index("{"):]
    parsed = json.loads(json_str)
    factor_matrices = {k: np.array(v) for k,v in parsed["factor_matrices"].items()}
    benchmark_info = {"wall_time":parsed["wall_time"]}
    loss_history = None

    return factor_matrices, loss_history, benchmark_info
