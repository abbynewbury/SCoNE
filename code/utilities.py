import numpy as np
import resource
import time
import os
import gc
from pathlib import Path
import shutil
import json
import pickle
import scone_tools.algorithms.SCoNE as SCoNE
import scone_tools.algorithms.MVBCWrapper as MVBCWrapper
import scone_tools.algorithms.RGWASWrapper as RGWASWrapper



def deploy_train_run(run_name,out_path,G=None,C=None,Z=None,reg_params=None,lambda_Gloss=None,
                    G_path='',C_path='',Z_path='',r_path='',rank=3, num_init=1, n_jobs=1,
                    write_all_init=False, write_all_init_path=''):

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


        
        algorithm_func_kwargs={"G":G, "C":C, "Z":Z,"rank":rank, "num_init":num_init, "init":"random", "n_jobs":n_jobs,
                        "alpha":reg_params['alpha'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'],"lambda_Gloss":lambda_Gloss,
                        "max_inner":20, "rho":0.1, "sigma":1e-4, "inner_ftol":1e-4,"max_ls":50,
                        "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                        "max_outer":300, "min_outer":10, "tol":1e-4,"post_hoc_rescale":True, "write_all_init":write_all_init, "write_all_init_path":write_all_init_path}
        factor_matrices, loss_function, benchmark_info = SCoNE.SCoNE_parallel(**algorithm_func_kwargs)

    elif run_name == 'MVBC':
        algorithm_func_kwargs = {"G_path":G_path,"C_path":C_path, "rank":rank,
                    "lambda_W":reg_params['lambda_W'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'], "r_path":r_path}
        factor_matrices, loss_function, benchmark_info = MVBCWrapper.MVBCWrapper(**algorithm_func_kwargs)

    elif run_name == 'RGWAS':
        algorithm_func_kwargs = {"r_path":r_path, "G_path":G_path,
                        "C_path":C_path, "Z_path":Z_path, "num_init":num_init,"rank":rank,
                        "write_all_init":write_all_init, "write_all_init_path":write_all_init_path}
        factor_matrices, loss_function, benchmark_info = RGWASWrapper.RGWASWrapper(**algorithm_func_kwargs)

    # write factor matrices and loss function to output path
    with open(f"{out_path}_loss_function.json", "w") as f: 
        json.dump(loss_function, f)
    with open(f"{out_path}_benchmark_info.json", "w") as f: 
        json.dump(benchmark_info, f)
    with open(f"{out_path}_factor_matrices.pkl", "wb") as f:
        pickle.dump(factor_matrices, f)


def deploy_test_run(run_name,out_path,G=None,C=None,Z=None,reg_params=None,lambda_Gloss=None,
                    H_G=None,H_C=None,U_G=None,U_C=None,rank=3,num_init=1,n_jobs=1,C_train=None,G_train=None,Z_train=None,
                    write_all_init=False, write_all_init_path=''):

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


    algorithm_func_kwargs={"G":G, "C":C, "Z":Z,"rank":rank, "num_init":num_init, "init":"random", "n_jobs":n_jobs,
                    "alpha":reg_params['alpha'], "lambda_H_G":reg_params['lambda_H_G'], "lambda_H_C":reg_params['lambda_H_C'],"lambda_Gloss":lambda_Gloss,
                    "max_inner":20, "rho":0.1, "sigma":1e-4, "inner_ftol":1e-4,"max_ls":50,
                    "G_loss_type":G_loss_type, "C_loss_type": C_loss_type,
                    "max_outer":300, "min_outer":10, "tol":1e-4,"post_hoc_rescale":False,"test":True,
                    "H_G":H_G, "H_C":H_C, "U_G":U_G, "U_C":U_C, "write_all_init":write_all_init, "write_all_init_path":write_all_init_path
                    }
    factor_matrices, loss_function, benchmark_info = SCoNE.SCoNE_parallel(**algorithm_func_kwargs)



    # write factor matrices and loss function to output path
    with open(f"{out_path}_loss_function.json", "w") as f: 
        json.dump(loss_function, f)
    with open(f"{out_path}_benchmark_info.json", "w") as f: 
        json.dump(benchmark_info, f)
    with open(f"{out_path}_factor_matrices.pkl", "wb") as f:
        pickle.dump(factor_matrices, f)


# +
def _call_kwargs_deploy_train_run(kw):
    kw = dict(kw)
    job_id = kw.pop("job_id")
    deploy_train_run(**kw)  
    return job_id

def _call_kwargs_deploy_test_run(kw):
    kw = dict(kw)
    job_id = kw.pop("job_id")
    deploy_test_run(**kw)  
    return job_id
