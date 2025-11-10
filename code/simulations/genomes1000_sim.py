import pandas as pd
import subprocess
from functools import reduce
import numpy as np
from plotnine import *
import os
import glob
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
import re
from scipy.stats import norm
import hashlib


def get_genetic_pcs(map_ped_filepath,output_dir,ndim=20):
    # make map/ped file into bfile
    plink_extract = f'''
    module load plink/1.9 && plink --file {map_ped_filepath} \
        --make-bed \
        --out {output_dir}/full_bfile
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

    result = subprocess.run(f'''module unload plink && module load flashpca\
                             && cd {output_dir} &&  flashpca --bfile {output_dir}/G --ndim {ndim}''', shell=True, capture_output=True, text=True, executable='/bin/bash')

    for f in glob.iglob(f"{output_dir}/full_bfile*"): # clean up
        os.remove(f)

def prep_1000genomes_bed_file(root_dir, output,subset_test=False):
    # change map and ped files to bed format
    # major allele set to A2 (If a binary fileset was originally loaded, --keep-allele-order forces the original A1/A2 allele encoding to be preserved; otherwise, the major allele is set to A2)
    plink_extract = f'''
    module load plink/1.9 && plink --file {root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05 \
        {"--thin-count 1000 --seed 42" if subset_test else ""}\
        --make-bed \
        --out {output}
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash") 

    # # make .raw file for G matrix later
    # plink_extract = f'''
    # module load plink/1.9 && plink --bfile {output} \
    #     --recode A \
    #     --out {output}
    # '''
    # result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

def read_in_igsr_samples(igsr_samples_filepath,bfile_path=None):
    # read in, subset to 2504, order correctly (if bfile path is not None)
    igsr_samples = pd.read_csv(igsr_samples_filepath,sep='\t')

    if bfile_path is not None:
        fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
        fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
        iid_order = fam_df['IID'].values
        igsr_samples = igsr_samples[igsr_samples['Sample name'].isin(iid_order)].copy()
        igsr_samples["Sample name"] = pd.Categorical(igsr_samples["Sample name"], categories=iid_order, ordered=True)
        igsr_samples = igsr_samples.sort_values("Sample name").reset_index(drop=True)
    igsr_samples["Superpopulation code"] = igsr_samples["Superpopulation code"].str.split(",").str[0] # chose first for sample with EUR,AFR superpopulation code
    igsr_samples.rename(columns={'Sample name':'IID'},inplace=True)
    igsr_samples['FID'] = igsr_samples['IID']

    return igsr_samples

def calculate_maf_by_superpop(igsr_samples_filepath,intermediate_file_dir,bfile_path,output):
      # Calculate allele frequencies in five superpopulations
      # generate superpopulation cluster file
      igsr_samples = read_in_igsr_samples(igsr_samples_filepath,bfile_path)
      igsr_samples[['FID','IID','Superpopulation code']].to_csv(f'{intermediate_file_dir}/superpop.clst',index=False,header=False,sep='\t')
      # calculate maf by superpop
      plink_freq = f''' module load plink/1.9 && 
      plink --bfile {bfile_path} \
            --freq \
            --within {intermediate_file_dir}/superpop.clst \
            --out {output}
      '''
      result = subprocess.run(plink_freq, shell=True, check=True, executable="/bin/bash")

      maf_by_superpop = pd.read_csv(f'{intermediate_file_dir}/maf_by_superpop.frq.strat',sep='\s+')
      return maf_by_superpop



def get_output_file_suffix(e,g_ps,c_ps,dataset):
    return f'e_{e}_g_ps_{g_ps}_c_ps_{c_ps}_dataset_{dataset}'

def rs(streams,stream_name):
        return int(streams[stream_name].integers(1, 2**31 - 1))

def seed_from(*xs):
    s = "|".join(f"{x:.8g}" if isinstance(x, float) else str(x) for x in xs)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)  # 32-bit int

def sun_generate_sim_data(bfile_path, maf_by_superpop_filepath,igsr_samples_filepath,
                          intermediate_file_dir,intermediate_file_suffix,output_dir,output_file_suffix,
                          g_ps,c_ps,g,e,M,num_clinical_assoc,num_markers, run_seed):
    '''
    Generate synthetic data similar to Sun et al. (Multi-view biclustering for genotype-phenotype association studies of complex diseases)
    using 1000 Genomes Phase 3 data. Use admixture files which contain 193634 markers with MAF>5% and 2504 individuals. 

    PARAMS:
    bfile_path: path to bfile for genetic data
    maf_by_superpop_filepath: pre-generated plink frq.strat file -- af stratified by superpopulation (from function calculate_maf_by_superpop)
    igsr_samples_filepath: filepath corresponding to igsr samples data on superpopulation (downloaded from https://www.internationalgenome.org/data-portal/sample on 08/20/25)
    admixture_fractions_filepath: path
    intermediate_file_dir: dir to write intermediate files to (when using plink for example)
    intermediate_file_suffix: such that if multiple simulations are created, each is distinctly defined
    output_file_suffix: such that if multiple simulations are created, each is distinctly defined - suffix for C and simulated data pkl file
    output_dir: where to write output genetic data matrix (X) in form of plink bfile, and clinical data matrix C
    M: number of clinical features (right now assuming all from one domain & all binary)
    g_ps: variable controlling how much population stratification is affecting genotype (value will determine quartile of AF variance (right now accepts 0.25 or 0.75)), or if 0 will pull from EUR superpopulation)
    c_ps: variable controlling the shift due to population stratification on phenotypic subgroup and clinical data matrix (value will determine range of uniform distribution shift Uniform(-a,a))
    (0-> pick SNPs in bottom 10% by allele frequency variance i.e. little pop. strat., 0.9-> pick SNPS in top 10% by allele frequency variance i.e. large pop. strat.)
    g: number of markers linked with subtype classification (if rij>int(0.4*markers_assoc) then subject i in subgroup j)
    e: relative effect that genetic variation contributed to the effect of the phenotype. e in [0,1]. (decreased e means higher level of disagreement between genotypic and phenotypic subgroups)
    num_clinical_assoc: number of clinical features associated with subtype classification (same for all subtypes)
    num_markers: total number of markers (i.e. G.shape[1]])
    run_seed: seed for run for reproducibility

    outputs:
    C: clinical data matrix (num samples x M)
    in pickle file (all simulation metadata):
    genetic_subgroups: genetic subgroup assignments for all individuals
    phenotypic_subgroups: phenotypic subgroup assignments for all individuals
    iid_order: IIDs in order (to align to clinical data matrix)
    markers_assoc_dict: for each genetic subgroup, IDs of genetic markers selected to be associated
    clinical_assoc_df: for each phenotypic subgroup, index of clinical variables selected to be associated (and their selected assoc. strength)

    '''
    assert num_clinical_assoc<M, "num_clinical_assoc cannot exceed M"
    parent_ss = np.random.SeedSequence(run_seed)
    names = ["env_noise", "extra_sub", "assoc", "poisson"]
    streams = {name: np.random.default_rng(ss) for name, ss in zip(names, parent_ss.spawn(len(names)))}
    streams["markers"] =  np.random.default_rng(seed_from("markers")) # fix null markers regardless of g_ps
    streams["marker_weights"] =  np.random.default_rng(seed_from("marker_weights")) # fix markers weights regardless of g_ps
    streams["ps_noise"] = np.random.default_rng(seed_from("ps_noise")) # fix superpopulation shift in a subset defined by c_ps
    streams["clinical_weights"] =  np.random.default_rng(seed_from("clinical_weights")) 
    
    # 1. Read in allele frequencies per 5 superpopulations to estimate af variance across groups OR pull from superpopulation EUR only
    maf_by_superpop = pd.read_csv(maf_by_superpop_filepath,sep='\s+')
    superpopulations = maf_by_superpop['CLST'].unique()
    maf_by_superpop = maf_by_superpop.pivot(index=['SNP'],columns='CLST',values='MAF').reset_index()
    igsr_samples = read_in_igsr_samples(igsr_samples_filepath,bfile_path)

    markers_assoc_df = [] # names & weights of the markers that are associated with each subgroup
    if g_ps !=0:
        assert g_ps in [0.25,0.75], f"right now code only takes top or bottom 25th percentile, value {g_ps} not accepted"
        # calculate weighted af variance
        w = igsr_samples['Superpopulation code'].value_counts().to_numpy(float)
        w_code_index = igsr_samples['Superpopulation code'].value_counts().index
        X = maf_by_superpop[w_code_index].to_numpy(float)  
        
        w = w / w.sum()
        mu = X @ w 
        maf_by_superpop['af_variance_weighted'] = ((X - mu[:, None])**2 * w[None, :]).sum(axis=1)
        fifth_percentile = q1 = maf_by_superpop['af_variance_weighted'].quantile(0.05) 
        q1 = maf_by_superpop['af_variance_weighted'].quantile(0.25)
        q3 = maf_by_superpop['af_variance_weighted'].quantile(0.75)
        # Mark bottom/top quartiles; leave middle as NaN 
        maf_by_superpop["af_var_quartile"] = np.select([maf_by_superpop['af_variance_weighted'] <= q1, maf_by_superpop['af_variance_weighted'] >= q3],[0.25, 0.75],default=np.nan)
        maf_by_superpop["null_pool"] = maf_by_superpop['af_variance_weighted'] <= fifth_percentile
        keep_samples = '' # keep all samples

        # select marker pool (num_markers markers where half are in right af_var_quartile and half are from null pool)
        marker_pool = maf_by_superpop[maf_by_superpop["null_pool"]].sample(n=num_markers//2, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist()
        linked_snp_pool = marker_pool.copy() # linked snps only drawn from null pool
        marker_pool.extend(maf_by_superpop[(maf_by_superpop['af_var_quartile']==g_ps)&(~maf_by_superpop['SNP'].isin(marker_pool))] 
                        .sample(n=num_markers//2, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist()) # & half from correct pool 


    else:
        eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
        # keep for european samples
        with open(f'{intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt','w') as f:
            for iid in eur_samples:
                f.write(iid + "\t" + iid + "\n")
        keep_samples = f'--keep {intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt'

        # select marker pool (where EUR MAF>5%) and pool from which to draw linked snps
        marker_pool = maf_by_superpop[maf_by_superpop['EUR']>=0.05].sample(n=num_markers, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist() 
        linked_snp_pool = marker_pool.copy()

    # select markers
    for genetic_subgroup in range(3):
        markers_assoc_df.append(pd.DataFrame({'genetic subgroup':genetic_subgroup, 'SNP': streams["markers"].choice(linked_snp_pool,size=g, replace=False).tolist()}))
    markers_assoc_df = pd.concat(markers_assoc_df)

    # 2. Generate genetic subgroups
    genetic_subgroups = []
    for genetic_subgroup in range(3):
        markers_assoc = markers_assoc_df[markers_assoc_df['genetic subgroup']==genetic_subgroup]['SNP'].values.tolist()
        assert len(set(markers_assoc))==len(markers_assoc) # make sure ped file has unique rows

        # extract selected markers
        with open(f'{intermediate_file_dir}/markers_assoc_g{genetic_subgroup}_{intermediate_file_suffix}.txt','w') as f:
            for snp in markers_assoc:
                f.write(snp + "\n")
        
        # extract select markers and get marker values for each individual (0 - no copies of minor allele, 1 - 1 copy of minor allele, 2 - 2 copies of minor allele)
        plink_extract = f'''
        module load plink/1.9 && plink --bfile {bfile_path} \
            --extract {intermediate_file_dir}/markers_assoc_g{genetic_subgroup}_{intermediate_file_suffix}.txt \
            {keep_samples} \
            --make-bed \
            --recode A \
            --out {intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}
        '''
        result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

        # read in to get genotype counts
        raw = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.raw',sep="\s+").rename(columns=lambda c: c.split('_')[0] if '_' in c else c)
        # assert they are all ≤ 0.5 (i.e., A1 is the minor allele)
        assert (raw[markers_assoc].sum(axis=0) / (2 * raw.shape[0]) <= 0.5).all(), "Some SNPs have A1 frequency > 0.5"
        r = raw.set_index('IID')[markers_assoc].sum(axis=1)
        genetic_subgroup_df = r.rename(f'r{genetic_subgroup}').to_frame().reset_index().rename(columns={'index':'IID'})
        genetic_subgroup_df[f'subgroup{genetic_subgroup}'] = (genetic_subgroup_df[f'r{genetic_subgroup}'] >= genetic_subgroup_df[f'r{genetic_subgroup}'].quantile(0.8)).astype(int)
        genetic_subgroups.append(genetic_subgroup_df)
    merge01=genetic_subgroups[0].merge(genetic_subgroups[1],on='IID') 
    genetic_subgroups = merge01.merge(genetic_subgroups[2],on='IID') 
    # remove overlapping samples in genetic subgroups 0 and 1
    subgroup_cols = ['subgroup0','subgroup1','subgroup2']
    # only keep iids in exactly one subgroup
    iids_in_one = genetic_subgroups.loc[genetic_subgroups[subgroup_cols].sum(axis=1) == 1, 'IID'].unique()
    genetic_subgroups = genetic_subgroups[genetic_subgroups['IID'].isin(iids_in_one)].copy()

    # 3. Generate phenotypic subgroups
    fam_df = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = [i for i in fam_df['IID'].values if i in iids_in_one]
    genetic_subgroups = genetic_subgroups.set_index('IID').loc[iid_order].reset_index() # make sure correct iid order
    gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(3): 
        g = genetic_subgroups[f'subgroup{phenotypic_subgroup}'].astype(int)
        g_standardized = (g - g.mean())/g.std()
        noise = streams["env_noise"].normal(loc=0,scale=1,size=len(g))
        latent = e*g_standardized + np.sqrt(1-e**2)*noise
        subgroup = (latent > np.quantile(latent, 0.8)).astype(int)
        phenotypic_subgroup_df = pd.DataFrame({'IID':iid_order})
        phenotypic_subgroup_df[f'subgroup{phenotypic_subgroup}'] = subgroup
        gi_phenotypic_subgroups.append(phenotypic_subgroup_df)
    merge01 = gi_phenotypic_subgroups[0].merge(gi_phenotypic_subgroups[1],on='IID') 
    gi_phenotypic_subgroups = merge01.merge(gi_phenotypic_subgroups[2],on='IID') 
    # only keep iids in exactly one subgroup
    iids_in_one = gi_phenotypic_subgroups.loc[gi_phenotypic_subgroups[subgroup_cols].sum(axis=1) == 1, 'IID'].unique()
    iid_order = [i for i in iid_order if i in iids_in_one] # remove overlap samples from genetic subgroups and iid order as well
    gi_phenotypic_subgroups =  gi_phenotypic_subgroups.set_index('IID').loc[iid_order].reset_index()
    genetic_subgroups = genetic_subgroups.set_index('IID').loc[iid_order].reset_index()
    extra_subgroups_size = gi_phenotypic_subgroups[subgroup_cols].sum().max()

    non_gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(3,5): 
        # randomly select extra_subgroups_size people
        randomly_selected = pd.Series(iid_order).sample(extra_subgroups_size,random_state=rs(streams,"extra_sub")).values.tolist() 
        phenotypic_subgroup_df = pd.DataFrame(iid_order,columns=['IID'])
        phenotypic_subgroup_df[f'subgroup{phenotypic_subgroup}'] = phenotypic_subgroup_df['IID'].isin(randomly_selected).astype(int)
        non_gi_phenotypic_subgroups.append(phenotypic_subgroup_df)
    non_gi_phenotypic_subgroups = non_gi_phenotypic_subgroups[0].merge(non_gi_phenotypic_subgroups[1],on='IID') 
    phenotypic_subgroups = gi_phenotypic_subgroups.merge(non_gi_phenotypic_subgroups,on='IID')

    # extract all markers & correct individuals for final G
    assert len(set(marker_pool)) == num_markers
    with open(f'{intermediate_file_dir}/allmarkers_{intermediate_file_suffix}.txt','w') as f:
        for snp in marker_pool:
            f.write(snp + "\n")
    with open(f'{intermediate_file_dir}/iids_{intermediate_file_suffix}.txt','w') as f:
        for iid in iid_order:
            f.write(iid + "\t" + iid + "\n")

    # extract all markers
    plink_extract = f'''
    module load plink/1.9 && plink --bfile {bfile_path} \
        --extract {intermediate_file_dir}/allmarkers_{intermediate_file_suffix}.txt \
        --keep {intermediate_file_dir}/iids_{intermediate_file_suffix}.txt \
        --make-bed \
        --recode A \
        --out {output_dir}/G_{output_file_suffix}
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")


    # 4. simulate M clinical features
    # start with baseline probabilities
    igsr_samples =  igsr_samples.set_index('IID').loc[iid_order].reset_index()
    Z = pd.get_dummies(igsr_samples['Superpopulation code']).to_numpy('float64')
    W = phenotypic_subgroups.set_index('IID').to_numpy('float64')
    # generate superpop shift for each feature
    U_C = np.clip(streams["ps_noise"].normal(loc=1, scale=0.1, size=(M, Z.shape[1])), 0, None)
    # pick num_clinical_assoc linked features for each subgroup
    H_C = np.zeros((M,4), float)
    for j in range(4):
        idx = streams["assoc"].choice(M, size=num_clinical_assoc, replace=False)
        H_C[idx, j] = np.clip(streams["clinical_weights"].normal(loc=0.6, scale=0.1, size=len(idx)), 0, None)
    shared_factors = streams['env_noise'].normal(size=(len(iid_order), 3))  # 3 correlated latent sources
    A = streams['env_noise'].normal(scale=0.1, size=(3, M))                 # loading matrix
    C = streams["poisson"].poisson(np.exp(0.1 + W@H_C.T + (c_ps)*Z@U_C.T + shared_factors@A + + streams["env_noise"].normal(0, 0.2, size=(len(iid_order), M))))
    clinical_assoc_df = pd.DataFrame(pd.DataFrame(H_C))

    # write C
    np.save(f"{output_dir}/C_{output_file_suffix}.npy", C)

    # write simulation metadata
    bundle = {
    "iid_order": iid_order,                          # list/array
    "genetic_subgroups": genetic_subgroups,          # list/array/Series
    "phenotypic_subgroups": phenotypic_subgroups,    # list/array/Series
    "markers_assoc": markers_assoc_df,             # dict: gen_subgroup -> [marker_id, ...]
    "clinical_assoc": clinical_assoc_df,              # pandas DataFrame
    "run_seed": run_seed                             # int
    }
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return genetic_subgroups, phenotypic_subgroups, C, iid_order, markers_assoc_df, clinical_assoc_df 
