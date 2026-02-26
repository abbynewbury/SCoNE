import numpy as np
import resource
import time
import os
import gc
from pathlib import Path
import shutil
import json
import pickle
import algorithms.SCoNE as SCoNE
import algorithms.MVBCWrapper as MVBCWrapper
import algorithms.RGWASWrapper as RGWASWrapper
import time


# Algorithm comparison functions
def profile_function(func, *args, mem_target='function', **kwargs):
    gc.collect()
    start_time = time.process_time()
    start_user_time = os.times().user
    if mem_target == 'function':
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elif mem_target == 'subprocess':
        start_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    result = func(*args, **kwargs)  # Run the function
    gc.collect()

    if mem_target == 'function':
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:  # 'subprocess'
        end_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    max_mem = (end_mem - start_mem) / 1024  # KB to MB
    cpu_time = time.process_time() - start_time
    user_time = os.times().user - start_user_time

    if isinstance(result, tuple):
        return (*result, max_mem, cpu_time, user_time)
    else:
        return result, max_mem, cpu_time, user_time


def deploy_train_run(run_name,out_path,G=None,C=None,Z=None,reg_params=None,lambda_Gloss=None,
                    G_path='',C_path='',Z_path='',r_path='',rank=3,num_init=1):

    if run_name in ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','HNMF(res)','CoNE','SCoNE','SCoNE(Fro)']:
        if 'SCoNE' not in run_name:
            reg_params = {'alpha':0,'lambda_H_G':0, 'lambda_H_C':0}
        if run_name=='HNMF(res)':
            C = np.maximum(C - Z @ np.linalg.lstsq(Z, C,rcond=None)[0],0)
            G = np.maximum(G - Z @ np.linalg.lstsq(Z, G,rcond=None)[0],0)
        
        # set G and C loss types
        if 'G-' in run_name: 
            G_loss_type,C_loss_type=('kl_div',None)
            C=None
            if 'NMF' in run_name:
                Z=None
        elif 'C-' in run_name: 
            G_loss_type,C_loss_type=(None,'kl_div')
            G=None
            if 'NMF' in run_name:
                Z=None
        elif run_name=='SCoNE(Fro)': G_loss_type,C_loss_type=('fro','fro')
        elif run_name=='HNMF':
            G_loss_type,C_loss_type=('kl_div','kl_div')
            Z=None
        elif run_name=='HNMF(res)':
            G_loss_type,C_loss_type=('fro','fro')
            Z=None           
        else: G_loss_type,C_loss_type=('kl_div','kl_div')


        
        algorithm_func_kwargs={"G":G, "C":C, "Z":Z,"rank":rank, "num_init":num_init, "init":"nndsvda",
                        "alpha":reg_params['alpha'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'],"lambda_Gloss":lambda_Gloss,
                        "max_inner":20, "rho":0.1, "sigma":1e-4, "inner_ftol":1e-4,"max_ls":50,
                        "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                        "max_outer":300, "min_outer":10, "tol":1e-4,"post_hoc_rescale":False}
        factor_matrices, loss_function = SCoNE.SCoNE_parallel(**algorithm_func_kwargs)

    elif run_name == 'MVBC':
        algorithm_func_kwargs = {"G_path":G_path,"C_path":C_path, "rank":rank,
                    "lambda_W":reg_params['lambda_W'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'], "r_path":r_path}
        factor_matrices, loss_function = MVBCWrapper.MVBCWrapper(**algorithm_func_kwargs)

    elif run_name == 'RGWAS':
        algorithm_func_kwargs = {"r_path":r_path, "G_path":G_path,
                        "C_path":C_path, "Z_path":Z_path, "num_init":num_init,"rank":rank}
        factor_matrices, loss_function = RGWASWrapper.RGWASWrapper(**algorithm_func_kwargs)

    # write factor matrices and loss function to output path
    with open(f"{out_path}_loss_function.json", "w") as f: 
        json.dump(loss_function, f)
    with open(f"{out_path}_factor_matrices.pkl", "wb") as f:
        pickle.dump(factor_matrices, f)


def deploy_test_run(run_name,out_path,G=None,C=None,Z=None,reg_params=None,lambda_Gloss=None,
                    H_G=None,H_C=None,U_G=None,U_C=None,rank=3,num_init=1,C_train=None,G_train=None,Z_train=None):

    if 'SCoNE' not in run_name:
        reg_params = {'alpha':0,'lambda_H_G':0, 'lambda_H_C':0}
    if run_name=='HNMF(res)':
        C = np.maximum(C - Z @ np.linalg.lstsq(Z_train, C_train,rcond=None)[0],0)
        G = np.maximum(G - Z @ np.linalg.lstsq(Z_train, G_train,rcond=None)[0],0)
        
    # set G and C loss types and test params
    if 'G-' in run_name: 
        G_loss_type,C_loss_type=('kl_div',None)
        C=None
        H_C=None
        U_C=None
        if 'NMF' in run_name:
            Z=None
    elif 'C-' in run_name: 
        G_loss_type,C_loss_type=(None,'kl_div')
        G=None
        H_G=None
        U_G=None
        if 'NMF' in run_name:
            Z=None
    elif run_name=='SCoNE(Fro)': G_loss_type,C_loss_type=('fro','fro')
    elif run_name=='HNMF':
        G_loss_type,C_loss_type=('kl_div','kl_div')
        Z=None
        U_G=None
        U_C=None
    elif run_name=='HNMF(res)':
        G_loss_type,C_loss_type=('fro','fro')
        Z=None           
        U_G=None
        U_C=None        
    else: G_loss_type,C_loss_type=('kl_div','kl_div')


    algorithm_func_kwargs={"G":G, "C":C, "Z":Z,"rank":rank, "num_init":num_init, "init":"random",
                    "alpha":reg_params['alpha'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'],"lambda_Gloss":lambda_Gloss,
                    "max_inner":20, "rho":0.1, "sigma":1e-4, "inner_ftol":1e-4,"max_ls":50,
                    "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                    "max_outer":300, "min_outer":10, "tol":1e-4,"post_hoc_rescale":False,"test":True,
                    "H_G":H_G, "H_C":H_C, "U_G":U_G, "U_C":U_C
                    }
    factor_matrices, loss_function = SCoNE.SCoNE_parallel(**algorithm_func_kwargs)



    # write factor matrices and loss function to output path
    with open(f"{out_path}_loss_function.json", "w") as f: 
        json.dump(loss_function, f)
    with open(f"{out_path}_factor_matrices.pkl", "wb") as f:
        pickle.dump(factor_matrices, f)


# +
def _call_kwargs_deploy_train_run(kw):
    kw = dict(kw)
    job_id = kw.pop("job_id")
    start = time.perf_counter()
    deploy_train_run(**kw)  
    return job_id, time.perf_counter() - start

def _call_kwargs_deploy_test_run(kw):
    kw = dict(kw)
    job_id = kw.pop("job_id")
    start = time.perf_counter()
    deploy_test_run(**kw)  
    return job_id, time.perf_counter() - start
