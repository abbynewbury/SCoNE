# ===== mlflow wrapper for alternating optimization =====
import mlflow
import os, pickle
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from pathlib import Path
import gc
import time
import resource

def train_with_mlflow(
    algorithm_func,         # expected to return factor_matrices, loss_history 
    # ^ (ASSUMES LOSS HISTORY KEYS HAVE VALUES ALL OF SAME LENGTH)
    algorithm_func_kwargs,  # dict of inputs to algorithm_func
    params,                 # param dict (including optimization params like tol)
    #state_dict,             # state dict defining model specifics (optimization params like tol)
    experiment_name, # name to identify dataset
    run_name,  # name to identify algorithm
    eval_fn = None, # for evaluating factor_matrices output (if none, no evaluation performed), should take factor_matrices, ground_truth as input
    # outputs metrics dict
    ground_truth = None, # ground truth for input into eval function
    artifact_dir: str = "artifacts",
    nested=False,
    confounding_matrix = None # for NMI with confounder/covariate matrix, rows here need to sum to 1
):
    """
    Wraps your algorithm function (defined in a .py file in algorithms folder) training loop with MLflow logging.

    Expects algorithm_func(...) to run all epochs and return:
      - loss
      - logs each entry in `loss_dict` as a metric series over epochs
      - optimization_dict: optimization options?

    Artifact dir used to store mlruns results
    """
    os.makedirs(artifact_dir, exist_ok=True)
    mlflow.set_tracking_uri("file:" + artifact_dir)
    exp = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=exp.experiment_id,run_name=run_name, nested=nested):
        # 1) Log parameters once
        mlflow.log_params(params)

        # 2) Run function  and log mem, cpu time, user time
        gc.collect()
        start_time = time.process_time()
        start_user_time = os.times().user
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        factor_matrices, loss_history = algorithm_func(**algorithm_func_kwargs)
        if factor_matrices is False: # failure reason stored in second output
            mlflow.log_param("failure_reason", loss_history)
            mlflow.end_run(status="FAILED")
            return 
        
        gc.collect()
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_mem = (end_mem - start_mem) / 1024  # KB to MB
        cpu_time = time.process_time() - start_time
        user_time = os.times().user - start_user_time
        mlflow.log_metrics({'max mem':max_mem, 'cpu time':cpu_time, 'user time':user_time})

    
        # ---------------------------------------------------------------

        # 3) Log training loss (and any extra info) with step=epoch
        assert len({len(v) for v in loss_history.values()}) == 1, f"Lengths differ: { {k: len(v) for k, v in loss_history.items()} }"
        num_epochs = [len(v) for v in loss_history.values()][0]
        for k,v in loss_history.items():
            for epoch in range(num_epochs):
                mlflow.log_metric(k, v[epoch], step=epoch)

        # 4) Evaluation
        if eval_fn:
            val_metrics = eval_fn(factor_matrices=factor_matrices,ground_truth=ground_truth,confounding_matrix=confounding_matrix)
            mlflow.log_metrics({k: v for k, v in val_metrics.items()})

        # 5) Log everything into final state as an artifact - for safe-keeping
        state = {}
        state["num_epochs"] = num_epochs
        for k, v in loss_history.items():
            state[k] = v
        state["method"] = run_name
        if eval_fn:
            for k, v in val_metrics.items():
                state[k] = v
        for k,v in params.items():
            state[k] = v
        for k,v in {'max mem':max_mem, 'cpu time':cpu_time, 'user time':user_time}.items():
            state[k] = v
        for k,v in factor_matrices.items():
            state[k] = v

        art_uri = mlflow.get_artifact_uri()       
        parsed = urlparse(art_uri)
        run_artifacts_dir = Path(parsed.path)
        ckpt_path = os.path.join(run_artifacts_dir, "final_state.pkl") 
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f)
        mlflow.log_artifact(ckpt_path, artifact_path="state_info")

        os.remove(ckpt_path)


        # 7) Log metadata
        mlflow.set_tag("trainer", algorithm_func)
        mlflow.set_tag("num_epochs", str(num_epochs))

        return state
