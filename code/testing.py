import pandas as pd

def is_sparse(run_name): return run_name in {'SCoNE','SCoNE(Fro)','MVBC'} # runs that have sparsity params


def run_one(variable_name, variable_range, output_dir):
    # ---- simulate data ----
    for variable in variable_range:
        sim_kwargs = {"n":500,"M_C":20,"num_genes":20,"noise":0.5,"ZU_weight": 0.5,"sparsity":0,"rho":0.8,"seed":0} 
        sim_kwargs[variable_name] = variable
        sim = simulate_views(**sim_kwargs) 
        # write G,C,Z to paths
        np.save(f'{tmp_folder}/G_{variable_name}_{variable}',sim["G"])
        np.save(f'{tmp_folder}/C_{variable_name}_{variable}',sim["C"])
        np.save(f'{tmp_folder}/Z_{variable_name}_{variable}',sim["Z"])

    # ---- config -----
    lambda_options = [0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100] 
    tuning_run_names = ['SCoNE' ,'SCoNE(Fro)','MVBC']
    testing_run_names = ['G-NMF','C-NMF','G-CoNE','C-CoNE','HNMF','CoNE','SCoNE','SCoNE(Fro)','RGWAS','MVBC']

    # ---- make compact DF ----
    plan = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_name
        for rn in tuning_run_names
        for split in ['training']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    new_rows = pd.DataFrame(
        [dict(variable=variable,
            run_name=rn,
            split=split,
            lambda_option=(lam if is_sparse(rn) else 0))
        for variable in variable_name
        for rn in tuning_run_names
        for split in ['tuning']
        for lam in (lambda_options if is_sparse(rn) else [0])]
    )
    plan = pd.concat([plan, new_rows], ignore_index=True)

    plan = plan[~((plan['run_name'].isin(['SCoNE','SCoNE(Fro)']))&(plan['lambda_option']==0))].copy()
    # generate job id and outpath
    # record file paths for results
    plan = plan.reset_index(drop=True)
    plan["job_id"] = plan.groupby(['variable','run_name','lambda_option']).ngroup()
    plan["out_path"] = "hypertension/models/"+ plan["split"] + "_" + plan["job_id"].astype(str)

    # deploy tuning runs
    tuning_args, tuning_rows = [], []
    for idx, row in plan[plan['split']=='tuning'].iterrows():
        G_path = f'{tmp_folder}/G_{variable_name}_{row.variable}'
        C_path = f'{tmp_folder}/C_{variable_name}_{row.variable}'
        Z_path = f'{tmp_folder}/Z_{variable_name}_{row.variable}'
        G = np.load(G_path)
        C = np.load(C_path)
        Z = np.load(Z_path)
        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,
            lambda_Gloss=(C.shape[1]/G.shape[1])*0.5, # since scale of C is only 0-1
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path="/usr/bin/Rscript", rank=3, num_init=10) 
        tuning_args.append(a)
        tuning_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=10, 
                                    lambda_Gloss=(C.shape[1]/G.shape[1])*0.5))
    tune_record = pd.DataFrame(tuning_rows)
    # ---- run TUNE ----
    computation_times_train = {}
    for tune_arg in tune_args:
        job_id, computation_time = _call_kwargs_deploy_train_run(train_arg)
        computation_times_train[job_id] = computation_time
    tune_record["computation_time"] = tune_record["job_id"].map(computation_times_train)
    tune_record.to_csv("hypertension/models/run_record_train.csv", index=False)
    print("done with tuning", flush=True)

    # deploy training runs
    training_args, training_rows = [], []
    for idx, row in plan[plan['split']=='training'].iterrows():
        G_path = f'{tmp_folder}/G_{variable_name}_{row.variable}'
        C_path = f'{tmp_folder}/C_{variable_name}_{row.variable}'
        Z_path = f'{tmp_folder}/Z_{variable_name}_{row.variable}'
        G = np.load(G_path)
        C = np.load(C_path)
        Z = np.load(Z_path)
        if row.lambda_option != 0: alpha = max(G.max(), C.max())**2
        else: alpha=0
        reg = {'lambda_W':row.lambda_option,'alpha':alpha,'lambda_H_G':row.lambda_option,'lambda_H_C':row.lambda_option}
        
        a = dict(job_id=row.job_id,run_name=row.run_name, out_path=row.out_path, G=G, C=C, Z=Z, reg_params=reg,
            lambda_Gloss=(C.shape[1]/G.shape[1])*0.5, # since scale of C is only 0-1
            G_path=G_path, C_path=C_path, Z_path=Z_path, r_path="/usr/bin/Rscript", rank=3, num_init=10) 
        training_args.append(a)
        training_rows.append(dict(out_path=row.out_path,variable=row.variable,
                                run_name=row.run_name,job_id=row.job_id,
                                lambda_option=row.lambda_option, alpha=reg['alpha'], lambda_W=reg['lambda_W'],
                                lambda_H_G=reg['lambda_H_G'],lambda_H_C=reg['lambda_H_C'], rank=3, num_init=10, 
                                    lambda_Gloss=(C.shape[1]/G.shape[1])*0.5))
    # ---- run TRAIN ----
    computation_times_train = {}
    for train_arg in train_args:
        job_id, computation_time = _call_kwargs_deploy_train_run(train_arg)
        computation_times_train[job_id] = computation_time
    train_record["computation_time"] = train_record["job_id"].map(computation_times_train)
    train_record.to_csv("hypertension/models/run_record_train.csv", index=False)
    print("done with training", flush=True)