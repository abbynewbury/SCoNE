#! /gpfs/commons/home/anewbury/miniconda/envs/jupyter/bin/python3
#SBATCH --job-name=RunRWASSim
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anewbury@nygenome.org
#SBATCH --output=output.txt
#SBATCH --error=errors.txt


import subprocess
from pathlib import Path
import argparse
import json
import sys
import threading
from threading import Semaphore, Thread
import time
import os


def monitor_job(job_id):
    """Monitor the specified job and release semaphore when it's completed."""
    while True:
        check = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        if job_id not in check.stdout:
            break  # Job is no longer in the queue
        time.sleep(30)  # Check every 30 seconds
    semaphore.release()

def submit_job(command):
    semaphore.acquire()
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stderr)
    print(result.stdout)
    result.check_returncode()
    job_id = result.stdout.strip().split()[-1]
    # Start a new thread to monitor the job
    monitor_thread = Thread(target=monitor_job, args=(job_id,))
    monitor_thread.start()

max_jobs = 40
semaphore = Semaphore(max_jobs)
unsupervised_pheno_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno'

# constant vars
N = 1000
S = 100
Q= 50
num_pops = 5
r_path = '/gpfs/commons/home/anewbury/miniconda/bin/Rscript'

# set base params
p_base= [0.5,0.5]
s_base = [25,25,50]
pge_base = 0.5
snp_hom_effects_base = 'small'
snps_af_range_base = [0.05,0.5]
mus_variance_base = 0.1
sim_id = 0

# VANILLA COMPARISON
command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
            '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
            '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base), 
            '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/vanilla_comparison', '--newton_cg', 'True','--bfgs', 'True',
            '--num_pops',5,'--r_path',r_path]
t = Thread(target=submit_job, args=(command,))
t.start()


# ORTHOGONAL COMPARISON
reg_params = [0,0.25,0.5,0.75,1,5,10,100]
command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
            '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
            '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base), 
            '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/orthogonal_comparison', '--reg_params', ' '.join(map(str,reg_params)),
             '--num_pops',5,'--r_path',r_path]
t = Thread(target=submit_job, args=(command,))
t.start()

# SENSITIVITY ANALYSIS FOR SIMULATION PARAMS
# also hold all others constant while tweaking one by a little to get data for sensitivity analysis
# vars for sensitivity analysis 
p_search = [[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1]]
s_search = [[25,50,25],[25,25,50],[20,40,40]]
pge_search = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
snp_hom_effects_search = ['small','large']
snps_af_range_search = [[0.05,0.5],[0.1,0.5],[0.2,0.5],[0.3,0.5],[0.4,0.5],[0.45,0.5]]
mus_variance_search = [0.1,0.2,0.3,0.4,0.5]

for p in p_search:
    print('iterating')
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
    '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

for s in s_search:
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
    '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

for pge in pge_search:
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge), '--snp_hom_effects', snp_hom_effects_base,
    '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

for snp_hom_effects in snp_hom_effects_search:
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects,
    '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance_base),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

for snps_af_range in snps_af_range_search:
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
    '--snps_af_range', ' '.join(map(str,snps_af_range)), '--mus_variance', str(mus_variance_base),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

for mus_variance in mus_variance_search:
    command = ['sbatch', f'{unsupervised_pheno_dir}/practice/rwas_simulations.py','--N',str(N),'--S',str(S),'--Q',str(Q), '--sim_id', str(sim_id), 
    '--p',' '.join(map(str,p_base)), '--s', ' '.join(map(str,s_base)), '--pge', str(pge_base), '--snp_hom_effects', snp_hom_effects_base,
    '--snps_af_range', ' '.join(map(str,snps_af_range_base)), '--mus_variance', str(mus_variance),
    '--simulation_results_path',f'{unsupervised_pheno_dir}/practice/simulation_results/sensitivity_sim_params','--reg_params', '0 5',
     '--num_pops',5,'--r_path',r_path]
    sim_id +=1
    t = Thread(target=submit_job, args=(command,))
    t.start()

print("All jobs have been submitted.")