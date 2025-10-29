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
    names = ["env_noise", "extra_sub", "assoc","poisson"]
    streams = {name: np.random.default_rng(ss) for name, ss in zip(names, parent_ss.spawn(len(names)))}
    streams["markers"] =  np.random.default_rng(seed_from("markers")) # fix null markers regardless of g_ps
    streams["ps_noise"] = np.random.default_rng(seed_from("ps_noise")) # fix superpopulation shift in a subset defined by c_ps

    # 1. Read in allele frequencies per 5 superpopulations to estimate af variance across groups OR pull from superpopulation EUR only
    maf_by_superpop = pd.read_csv(maf_by_superpop_filepath,sep='\s+')
    superpopulations = maf_by_superpop['CLST'].unique()
    maf_by_superpop = maf_by_superpop.pivot(index=['SNP'],columns='CLST',values='MAF').reset_index()
    igsr_samples = read_in_igsr_samples(igsr_samples_filepath,bfile_path)

    markers_assoc_dict = {} # names of the markers that are associated with each subgroup
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
        marker_pool.extend(maf_by_superpop[(maf_by_superpop['af_var_quartile']==g_ps)&(~maf_by_superpop['SNP'].isin(marker_pool))] 
                        .sample(n=num_markers//2, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist()) # & half from correct pool 

        # select markers
        for genetic_subgroup in range(2):
            markers_assoc = streams["markers"].choice(marker_pool,size=g, replace=False).tolist()
            markers_assoc_dict[genetic_subgroup] = markers_assoc

    else:
        eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
        # keep for european samples
        with open(f'{intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt','w') as f:
            for iid in eur_samples:
                f.write(iid + "\t" + iid + "\n")
        keep_samples = f'--keep {intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt'

        # select marker pool (where EUR MAF>5%)
        marker_pool = maf_by_superpop[maf_by_superpop['EUR']>=0.05].sample(n=num_markers, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist() 

        # select markers
        for genetic_subgroup in range(2):
            markers_assoc = streams["markers"].choice(marker_pool,size=g, replace=False).tolist()
            markers_assoc_dict[genetic_subgroup] = markers_assoc

    # 2. Generate genetic subgroups
    genetic_subgroups = []
    for genetic_subgroup in range(2):
        markers_assoc = markers_assoc_dict[genetic_subgroup]
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
        raw = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.raw',sep="\s+")
        geno = raw.drop(columns=['FID','IID','PAT','MAT','SEX','PHENOTYPE'])
        # assert they are all ≤ 0.5 (i.e., A1 is the minor allele)
        assert (geno.sum(axis=0) / (2 * geno.shape[0]) <= 0.5).all(), "Some SNPs have A1 frequency > 0.5"
        geno_cols = [col for col in raw.columns if not col in ['FID','IID','PAT','MAT','SEX','PHENOTYPE']]
        assert len(geno_cols) == g
        #raw[geno_cols] = (raw[geno_cols] > 0).astype(int) # recode s.t. values 1 and 2 map to 1 - keep columns in 0/1/2
        genetic_subgroup_df = raw.set_index('IID')[geno_cols].sum(axis=1).reset_index(name=f'r')
        genetic_subgroup_df[f'subgroup{genetic_subgroup}'] = (genetic_subgroup_df['r'] > genetic_subgroup_df['r'].quantile(0.8)).astype(int)
        genetic_subgroups.append(genetic_subgroup_df[['IID',f'subgroup{genetic_subgroup}']])
    genetic_subgroups = genetic_subgroups[0].merge(genetic_subgroups[1],on='IID') 
    # remove overlapping samples in genetic subgroups 0 and 1
    overlap_iids = genetic_subgroups[(genetic_subgroups['subgroup0']==1)&(genetic_subgroups['subgroup1']==1)]['IID'].unique()
    genetic_subgroups = genetic_subgroups[~genetic_subgroups['IID'].isin(overlap_iids)].copy()


    # 3. Generate phenotypic subgroups
    fam_df = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = [i for i in fam_df['IID'].values if i not in overlap_iids]
    genetic_subgroups = genetic_subgroups.set_index('IID').loc[iid_order].reset_index() # make sure correct iid order
    gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(2): 
        g = genetic_subgroups[f'subgroup{phenotypic_subgroup}'].astype(int)
        g_standardized = (g - g.mean())/g.std()
        noise = streams["env_noise"].normal(loc=0,scale=1,size=len(g))
        latent = e*g_standardized + np.sqrt(1-e**2)*noise
        subgroup = (latent > np.quantile(latent, 0.8)).astype(int)
        phenotypic_subgroup_df = pd.DataFrame({'IID':iid_order})
        phenotypic_subgroup_df[f'subgroup{phenotypic_subgroup}'] = subgroup
        gi_phenotypic_subgroups.append(phenotypic_subgroup_df)
    gi_phenotypic_subgroups = gi_phenotypic_subgroups[0].merge(gi_phenotypic_subgroups[1],on='IID') 
    # remove overlapping samples in phenotypic subgroups 0 and 1
    overlap_iids = gi_phenotypic_subgroups[(gi_phenotypic_subgroups['subgroup0']==1) & (gi_phenotypic_subgroups['subgroup1']==1)]['IID'].unique()
    iid_order = [i for i in iid_order if i not in overlap_iids] # remove overlap samples from genetic subgroups and iid order as well
    gi_phenotypic_subgroups =  gi_phenotypic_subgroups.set_index('IID').loc[iid_order].reset_index()
    genetic_subgroups = genetic_subgroups.set_index('IID').loc[iid_order].reset_index()
    extra_subgroups_size = gi_phenotypic_subgroups[['subgroup0','subgroup1']].sum().max()

    non_gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(2,4): 
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
    U_C = streams["ps_noise"].gamma(shape=1.0, scale=1.0, size=(M, Z.shape[1]))  
    # pick num_clinical_assoc linked features for each subgroup
    H_C = np.zeros((M,4), int)
    for j in range(4):
        idx = streams["assoc"].choice(M, size=num_clinical_assoc, replace=False)
        H_C[idx, j] = 1
    C = streams["poisson"].poisson(np.exp(0.1 + W@H_C.T + (c_ps)*Z@U_C.T))
    clinical_assoc_df = pd.DataFrame(pd.DataFrame(H_C))

    # write C
    np.save(f"{output_dir}/C_{output_file_suffix}.npy", C)

    # write simulation metadata
    bundle = {
    "iid_order": iid_order,                          # list/array
    "genetic_subgroups": genetic_subgroups,          # list/array/Series
    "phenotypic_subgroups": phenotypic_subgroups,    # list/array/Series
    "markers_assoc": markers_assoc_dict,             # dict: gen_subgroup -> [marker_id, ...]
    "clinical_assoc": clinical_assoc_df,              # pandas DataFrame
    "run_seed": run_seed                             # int
    }
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return genetic_subgroups, phenotypic_subgroups, C, iid_order, markers_assoc_dict, clinical_assoc_df 


# old function from 10/22
def old_sun_generate_sim_data(bfile_path, maf_by_superpop_filepath,igsr_samples_filepath,
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
    streams["markers"] =  np.random.default_rng(int(g_ps*100)) # fix markers within a marker subset defined by g_ps
    streams["ps_noise"] = np.random.default_rng(int(c_ps*100)) # fix superpopulation shift in a subset defined by c_ps

    # 1. Read in allele frequencies per 5 superpopulations to estimate af variance across groups OR pull from superpopulation EUR only
    maf_by_superpop = pd.read_csv(maf_by_superpop_filepath,sep='\s+')
    superpopulations = maf_by_superpop['CLST'].unique()
    maf_by_superpop = maf_by_superpop.pivot(index=['SNP'],columns='CLST',values='MAF').reset_index()
    igsr_samples = read_in_igsr_samples(igsr_samples_filepath,bfile_path)

    markers_assoc_dict = {} # names of the markers that are associated with each subgroup
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
        marker_pool.extend(maf_by_superpop[(maf_by_superpop['af_var_quartile']==g_ps)&(~maf_by_superpop['SNP'].isin(marker_pool))] 
                        .sample(n=num_markers//2, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist()) # & half from correct pool 

        # select markers
        for genetic_subgroup in range(2):
            markers_assoc = streams["markers"].choice(marker_pool,size=g, replace=False).tolist()
            markers_assoc_dict[genetic_subgroup] = markers_assoc

    else:
        eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
        # keep for european samples
        with open(f'{intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt','w') as f:
            for iid in eur_samples:
                f.write(iid + "\t" + iid + "\n")
        keep_samples = f'--keep {intermediate_file_dir}/eursamples_{intermediate_file_suffix}.txt'

        # select marker pool (where EUR MAF>5%)
        marker_pool = maf_by_superpop[maf_by_superpop['EUR']>=0.05].sample(n=num_markers, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist() 

        # select markers
        for genetic_subgroup in range(2):
            markers_assoc = streams["markers"].choice(marker_pool,size=g, replace=False).tolist()
            markers_assoc_dict[genetic_subgroup] = markers_assoc
    
    # 2. Generate genetic subgroups
    genetic_subgroups = []
    for genetic_subgroup in range(2):
        markers_assoc = markers_assoc_dict[genetic_subgroup]
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
        raw = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.raw',sep="\s+")
        geno = raw.drop(columns=['FID','IID','PAT','MAT','SEX','PHENOTYPE'])
        # assert they are all ≤ 0.5 (i.e., A1 is the minor allele)
        assert (geno.sum(axis=0) / (2 * geno.shape[0]) <= 0.5).all(), "Some SNPs have A1 frequency > 0.5"
        geno_cols = [col for col in raw.columns if not col in ['FID','IID','PAT','MAT','SEX','PHENOTYPE']]
        assert len(geno_cols) == g
        #raw[geno_cols] = (raw[geno_cols] > 0).astype(int) # recode s.t. values 1 and 2 map to 1 - keep columns in 0/1/2
        genetic_subgroup_df = raw.set_index('IID')[geno_cols].sum(axis=1).reset_index(name=f'r')
        genetic_subgroup_df['genetic_subgroup'] = genetic_subgroup
        genetic_subgroup_df[f'subgroup'] = genetic_subgroup_df['r'] > genetic_subgroup_df['r'].quantile(0.9)
        genetic_subgroups.append(genetic_subgroup_df)
    genetic_subgroups = pd.concat(genetic_subgroups)
    # remove overlapping samples in genetic subgroups 0 and 1
    mask = genetic_subgroups['subgroup'] & genetic_subgroups['genetic_subgroup'].isin([0, 1])
    overlap_iids = (genetic_subgroups.loc[mask]
                  .groupby('IID')['genetic_subgroup']
                  .nunique()
                  .pipe(lambda s: s[s > 1]).index)
    genetic_subgroups = genetic_subgroups[~genetic_subgroups['IID'].isin(overlap_iids)].copy()


    # 3. Generate phenotypic subgroups
    fam_df = pd.read_csv(f'{intermediate_file_dir}/subset_markers_g{genetic_subgroup}_{intermediate_file_suffix}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = [i for i in fam_df['IID'].values if i not in overlap_iids]
    gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(2): 
        phenotypic_subgroup_df = genetic_subgroups[genetic_subgroups['genetic_subgroup']==phenotypic_subgroup][['IID','r']].copy()
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        phenotypic_subgroup_df = phenotypic_subgroup_df.merge(igsr_samples[['IID','Superpopulation code']],on='IID',how='inner')
        # corresponding genetic subgroup value for r
        phenotypic_subgroup_df['diff'] = phenotypic_subgroup_df['r'].quantile(0.9) - phenotypic_subgroup_df['r']
        phenotypic_subgroup_df['diff_z'] = (phenotypic_subgroup_df['diff']) / phenotypic_subgroup_df['diff'].std()# standardize scale
        phenotypic_subgroup_df['subgroup'] = streams["env_noise"].normal(loc=0,scale=1,size=phenotypic_subgroup_df.shape[0])> phenotypic_subgroup_df['diff_z']*e
        gi_phenotypic_subgroups.append(phenotypic_subgroup_df)
    gi_phenotypic_subgroups = pd.concat(gi_phenotypic_subgroups)
    # remove overlapping samples in phenotypic subgroups 0 and 1
    mask = gi_phenotypic_subgroups['subgroup']
    overlap_iids = (gi_phenotypic_subgroups.loc[mask]
                  .groupby('IID')['phenotypic_subgroup']
                  .nunique()
                  .pipe(lambda s: s[s > 1]).index)
    gi_phenotypic_subgroups = gi_phenotypic_subgroups[~gi_phenotypic_subgroups['IID'].isin(overlap_iids)].copy()
    extra_subgroups_size = 2*max(gi_phenotypic_subgroups[gi_phenotypic_subgroups['subgroup']]['phenotypic_subgroup'].value_counts()) # is max of pheno subgroup 0 or 1 # TODO: change back to not everyone
    # remove overlap samples from genetic subgroups and iid order as well
    genetic_subgroups = genetic_subgroups[~genetic_subgroups['IID'].isin(overlap_iids)].copy()
    iid_order = [i for i in iid_order if i not in overlap_iids]
    non_gi_phenotypic_subgroups = []
    for phenotypic_subgroup in range(2,4): 
        # randomly select extra_subgroups_size people
        randomly_selected = pd.Series(iid_order).sample(extra_subgroups_size,random_state=rs(streams,"extra_sub")).values.tolist() 
        phenotypic_subgroup_df = pd.DataFrame(iid_order,columns=['IID'])
        phenotypic_subgroup_df['r'] = None
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        phenotypic_subgroup_df['subgroup'] = phenotypic_subgroup_df['IID'].isin(randomly_selected)
        non_gi_phenotypic_subgroups.append(phenotypic_subgroup_df)
    non_gi_phenotypic_subgroups = pd.concat(non_gi_phenotypic_subgroups)
    phenotypic_subgroups = pd.concat([gi_phenotypic_subgroups,non_gi_phenotypic_subgroups])

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


    # 4. simulate M binary clinical features
    # start with baseline probabilities
    sp2bump = {sp: streams["ps_noise"].uniform(-c_ps, c_ps)
        for sp in igsr_samples['Superpopulation code'].unique()}
    subj_bump = (igsr_samples
                .set_index('IID')['Superpopulation code']
                .reindex(iid_order).map(sp2bump).to_numpy())  # shape (n,)

    clip01 = lambda x: np.clip(x, 1e-6, None)

    # baseline (add ps bump, broadcast across M)
    C = streams["poisson"].poisson(clip01(0.1 + subj_bump)[:, None], size=(len(iid_order), M))
    clinical_assoc_df_rows = [] # index of clinical vars that are associated (and their strength)
    for phenotypic_subgroup in phenotypic_subgroups['phenotypic_subgroup'].unique():
        # index of randomly chosen, associated clinical variables 
        assoc_idx = streams["assoc"].choice(M, size=num_clinical_assoc, replace=False)  
        n1 = num_clinical_assoc // 3 # 1/3 
        n2 = 2 * num_clinical_assoc // 3 # 1/3 
        # get those who are in the subgroup
        mask = ((phenotypic_subgroups["phenotypic_subgroup"] == phenotypic_subgroup) & (phenotypic_subgroups["subgroup"]))
        subj_ids = phenotypic_subgroups.loc[mask, "IID"].unique()
        # map subject IDs to row indices 
        row_idx = [i for i, iid in enumerate(iid_order) if iid in subj_ids]
        b = subj_bump[row_idx][:, None] # per subject bump from population structure
        C[np.ix_(row_idx, assoc_idx[:n1])] = streams["poisson"].poisson(clip01(0.6+b), size=(len(row_idx), n1))
        C[np.ix_(row_idx, assoc_idx[n1:n2])] = streams["poisson"].poisson(clip01(0.5+b), size=(len(row_idx), n2-n1))
        C[np.ix_(row_idx, assoc_idx[n2:])] = streams["poisson"].poisson(clip01(0.4+b), size=(len(row_idx), len(assoc_idx)-n2))
        clinical_assoc_df_rows += [
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.6, "indices": assoc_idx[:n1]},
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.5, "indices": assoc_idx[n1:n2]},
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.4, "indices": assoc_idx[n2:]},
        ]
    clinical_assoc_df = pd.DataFrame(clinical_assoc_df_rows)

    # write C
    np.save(f"{output_dir}/C_{output_file_suffix}.npy", C)

    # write simulation metadata
    bundle = {
    "iid_order": iid_order,                          # list/array
    "genetic_subgroups": genetic_subgroups,          # list/array/Series
    "phenotypic_subgroups": phenotypic_subgroups,    # list/array/Series
    "markers_assoc": markers_assoc_dict,             # dict: gen_subgroup -> [marker_id, ...]
    "clinical_assoc": clinical_assoc_df,              # pandas DataFrame
    "run_seed": run_seed                             # int
    }
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return genetic_subgroups, phenotypic_subgroups, C, iid_order, markers_assoc_dict, clinical_assoc_df 

### FUNCTIONS FOR DATA SIM EVALUATION:

# PCA RELATED
def clean_join(values):
    # remove empty strings
    vals = sorted(set(v for v in values if v != ''), key=lambda x: int(x))
    return ','.join(vals)

def generate_pca_plot(mode, var_list, color_col, color_label, output_dir, graph_dir, igsr_samples_filepath=None): 
    '''Generate UMAP plot of C across different values of variable spcified in 'mode' 
    (values in var_list), colored by color_df (which must have a column IID)
    
    output dir and output suffix define where to find simulated data files
    '''
    plot_dfs = []
    for var in var_list:
        if mode == 'ps':
            output_file_suffix = get_output_file_suffix(ps=var,e=0.5,dataset=0,g=10) # choose first dataset and 10 markers assoc for vis purposes
            with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            color_df = read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/G')
        else:
            assert mode=='e', "only works with modes ps and e so far"
            output_file_suffix = get_output_file_suffix(ps=True,e=var,dataset=0,g=10) # choose first dataset and 10 markers assoc for vis purposes
            with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            genetic_subgroups = simulation_metadata['genetic_subgroups']
            genetic_subgroups['subgroup_value'] = np.where(genetic_subgroups['subgroup'],genetic_subgroups['genetic_subgroup'],'') # change this to one label per person
            genetic_subgroups_concat = genetic_subgroups.groupby('IID')['subgroup_value'].apply(clean_join).reset_index()
            genetic_subgroups_concat["IID"] = pd.Categorical(genetic_subgroups_concat["IID"], categories=iid_order, ordered=True)
            color_df = genetic_subgroups_concat.copy()
        C = np.load(f'{output_dir}/C_{output_file_suffix}.npy')# pick e=0.5
        C_scaled = StandardScaler().fit_transform(C) # standardize matrix before PCA

        pca = PCA(n_components=2, random_state=0)     # choose target dims
        embedding = pca.fit_transform(C_scaled) 

        plot_df = (pd.DataFrame(embedding, columns=["PC1", "PC2"], index=iid_order).
                    rename_axis("IID").reset_index().merge(color_df[['IID',color_col]], on='IID',how='inner'))
        
        assert plot_df[plot_df[color_col].isna()].shape[0] == 0
        plot_df[mode] = var
        plot_dfs.append(plot_df)
    plot_dfs = pd.concat(plot_dfs)

    # --- Plot with plotnine ---
    if mode == 'ps':
        title = r"PCA of clinical data at different $p_s$ levels"
    else:
        title = r"PCA of clinical data at different e levels"
    p = (
        ggplot(plot_dfs, aes("PC1", "PC2", color=color_col))
        + geom_point(alpha=0.5, size=2)
        + labs(title=title, color=color_label)
        + facet_wrap(f'~{mode}',ncol=2,scales='free')
        + theme_minimal()
        + theme(legend_title=element_text(size=9))
    )
    p.save(f'{graph_dir}/pca_clinical_{mode}.pdf',dpi=300)

# GWAS RELATED

# gwas setup
def generate_phenotype_file(simulation_metadata_path, phenotypic_subgroup, covar_filepath, phenotype_file_plink, phenotype_file_saige):
    # write plink phenotype file
    with open(simulation_metadata_path, "rb") as f:
        simulation_metadata = pickle.load(f)
    phenotypic_subgroups = simulation_metadata['phenotypic_subgroups']

    # write plink phenotype file
    pheno = phenotypic_subgroups[phenotypic_subgroups['phenotypic_subgroup']==phenotypic_subgroup][['IID','subgroup']].rename(columns={'subgroup':'Phenotype'})
    pheno['FID'] = pheno['IID']
    pheno['Phenotype'] = pheno['Phenotype'].astype(int)
    pheno[['FID','IID','Phenotype']].set_index('FID').to_csv(phenotype_file_plink)

    # write saige phenotype file (this includes covariates)
    pheno.drop('FID',axis=1,inplace=True)
    covar_df = pd.read_csv(covar_filepath).drop('FID',axis=1)
    saige_phenoFile = pheno.merge(covar_df,how='inner',on='IID').set_index('IID')
    assert (set(saige_phenoFile.index) == set(pheno['IID'])) and (set(pheno['IID'])==set(covar_df[covar_df['IID'].isin(saige_phenoFile.index)]['IID']))
    saige_phenoFile.to_csv(phenotype_file_saige,sep='\t')

# plink gwas

def run_plink_gwas(bfile, covariate_file, phenotype_file, out):
    if covariate_file is None:
        covar_section = ''
        covar_flag = 'allow-no-covars'
    else:
        covar_section = f'--covar {covariate_file} --covar-variance-standardize'
        covar_flag = ''
    result = subprocess.run(f'module unload plink && module load plink/2.0a5.13 && plink --bfile {bfile}\
                        {covar_section}\
                        --pheno {phenotype_file}\
                        --glm omit-ref {covar_flag}\
                        --out {out}\
                        --1 --no-pheno', shell=True, capture_output=True, text=True, executable='/bin/bash')
    result.check_returncode()

    # write plink results to parquet for quicker analysis - write all results to plink
    with open(f'{out}.log','r') as f:
        file = f.read()
        assert "End time" in file, f"plink ended with errors for {out}"
    plink_results = pd.read_csv(f'{out}.Phenotype.glm.logistic.hybrid',sep='\t')
    plink_results = plink_results[(plink_results['TEST']=='ADD')&(plink_results['ERRCODE']=='.')].copy() # only write SNP effect size data
    plink_results["OR"] = pd.to_numeric(plink_results["OR"])
    plink_results.to_parquet(f'{out}.parquet', engine='pyarrow') # export to parquet format for quicker lookup later on
    # clean up for storage space
    os.remove(f'{out}.Phenotype.glm.logistic.hybrid')

# saige gwas
def split_plink_bfile(bfile,out_path):
    chrs_absent = []
    for chr in range(1,23):
        result = subprocess.run(f'''module load plink/1.9 && plink --bfile {bfile} \
          --chr {chr} \
          --make-bed \
          --out {out_path}_{chr}''', shell=True, capture_output=True, text=True, executable='/bin/bash')
        if result.returncode != 0:
            msg = (result.stdout + result.stderr).lower()
            # tolerate common "no variants" phrasings from PLINK
            no_var = any(s in msg for s in [
                "all variants excluded", "no valid variants", "all variants removed", "contains zero variants"
            ])
            if no_var:
                print(f"chr {chr}: no variants; skipping")
                chrs_absent.append(chr)
                [os.remove(f) for f in glob.glob(f"{out_path}_{chr}*")]
                continue
            # otherwise, surface the error
            sys.stderr.write(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, "plink split bfile", output=result.stdout, stderr=result.stderr)
    return [i for i in range(1,23) if i not in chrs_absent]
# step 1: run null GRM for each phenotype
def run_step1(plinkFile, phenoFile, out):
    saige_phenoFile = pd.read_csv(phenoFile,sep='\t')
    # run null model
    result = subprocess.run(f"module load R/4.3.3 \
                            && Rscript $(echo $CMAKE_PREFIX_PATH | tr ':' '\n' | grep R/4.3.3)/SAIGE/extdata/step1_fitNULLGLMM.R \
                            --useSparseGRMtoFitNULL=FALSE \
                            --plinkFile={plinkFile} \
                            --phenoFile={phenoFile}\
                            --skipVarianceRatioEstimation=FALSE \
                            --phenoCol=Phenotype \
                            --covarColList={','.join([i for i in saige_phenoFile.columns if i not in ['Phenotype','IID']])} \
                            --qCovarColList=Sex \
                            --sampleIDColinphenoFile=IID \
                            --traitType=binary \
                            --LOCO=TRUE \
                            --outputPrefix={out} \
                            --nThreads=16 \
                            --isCovariateOffset=FALSE \
                            --IsOverwriteVarianceRatioFile=TRUE", shell=True, capture_output=True, text=True, executable='/bin/bash')
    result.check_returncode()

def active_count(job_ids):
    if not job_ids: return 0
    cmd = f"squeue -h -j {','.join(job_ids)}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # count non-empty lines
    return sum(1 for line in r.stdout.splitlines() if line.strip())

def submit_step2_job(chrs_present, bfile, GMMATmodelFile, varianceRatioFile, intermediate_saige_dir, out):
    # run association tests LOCO
    job_ids = []
    for chr in chrs_present:
        while active_count(job_ids) >= 5:
            time.sleep(30)
        job_name = f"{os.path.basename(out)}_chr_{chr}"
        slurm_script = f'{intermediate_saige_dir}/{job_name}.sh'
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={intermediate_saige_dir}/{job_name}.out
#SBATCH --error={intermediate_saige_dir}/{job_name}.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module load R/4.3.3 
Rscript $(echo $CMAKE_PREFIX_PATH | tr ':' '\n' | grep R/4.3.3)/SAIGE/extdata/step2_SPAtests.R \\
--bedFile={bfile}_{chr}.bed \\
--bimFile={bfile}_{chr}.bim \\
--famFile={bfile}_{chr}.fam \\
--AlleleOrder=alt-first \\
--SAIGEOutputFile={out}_chr_{chr} \\
--GMMATmodelFile={GMMATmodelFile} \\
--varianceRatioFile={varianceRatioFile} \\
--LOCO=TRUE \\
--chrom={chr} \\
--is_Firth_beta=TRUE    \\
--pCutoffforFirth=0.01 \\
    """

        # Write script to file
        with open(slurm_script, 'w') as f:
            f.write(slurm_content)

        # Submit the job
        result = subprocess.run(f"sbatch {slurm_script}", shell=True, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
    
    # wait for all jobs to complete
    while active_count(job_ids) > 0:
        time.sleep(30)

    # consolidate saige results
    saige_results = []
    for chr in chrs_present:
        saige_chr_results = pd.read_csv(f'{out}_chr_{chr}',sep='\t')
        saige_results.append(saige_chr_results)
    saige_results = pd.concat(saige_results)
    saige_results.to_csv(f'{out}',index=False)

    # remove excess files
    for f in glob.iglob(f"{intermediate_saige_dir}/{os.path.basename(out)}*"): # clean up
        os.remove(f)
    for f in glob.iglob(f"{out}_chr_*"): # clean up
        os.remove(f)
    


def run_saige(plinkFile,phenoFile,intermediate_saige_dir,output_dir,phenotypic_subgroup,output_file_suffix):
    out_suffix = f'PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_SAIGE'
    run_step1(plinkFile=plinkFile, phenoFile=phenoFile,
              out=f'{intermediate_saige_dir}/{out_suffix}')
    
    # split plink bfile by chr (for SAIGE LOCO) and record which chrs are present
    chrs_present = split_plink_bfile(f'{output_dir}/G_{output_file_suffix}',f'{intermediate_saige_dir}/G_{output_file_suffix}')

    # run association tests LOCO
    GMMATmodelFile = f'{intermediate_saige_dir}/{out_suffix}.rda'
    varianceRatioFile = f'{intermediate_saige_dir}/{out_suffix}.varianceRatio.txt'
    # runs for each chr then consolidates into out file
    submit_step2_job(chrs_present=chrs_present, bfile=f'{intermediate_saige_dir}/G_{output_file_suffix}', GMMATmodelFile=GMMATmodelFile, varianceRatioFile=varianceRatioFile,
                    intermediate_saige_dir=intermediate_saige_dir, 
                      out=f'{output_dir}/GWAS_RESULTS/{out_suffix}')

    # make sure ran assoc. testing on all desired snps
    bim_df = pd.read_csv(f'{output_dir}/G_{output_file_suffix}.bim',sep='\s+',header=None,names=['CHR','SNP','CM','POS','A1','A2']).reset_index(drop=True)
    saige_results = pd.read_csv(f'{output_dir}/GWAS_RESULTS/{out_suffix}')
    assert set(bim_df['SNP'].values) == set(saige_results['MarkerID'].values), f'bim df snps != saige assoc. test snps for saige {output_dir}/GWAS_RESULTS/{out_suffix}; bim {output_dir}/G_{output_file_suffix}'

def consolidate_files(plink_nocovs_filepath, plink_covs_filepath, saige_filepath, out):
    # merge all into one long format parquet
    plink_results_nocovs = pd.read_parquet(plink_nocovs_filepath)
    assert all(plink_results_nocovs['A1'] == plink_results_nocovs['ALT'])
    plink_results_nocovs['BETA'] = np.log(plink_results_nocovs['OR']) 
    plink_results_nocovs = plink_results_nocovs.rename(columns={'LOG(OR)_SE':'SE','Z_STAT':'Tstat'})[['#CHROM','POS','ID','REF','ALT','BETA','SE','Tstat','P']].copy()
    plink_results_nocovs['assoc_test'] = 'LR'

    plink_results_covs = pd.read_parquet(plink_covs_filepath)
    assert all(plink_results_covs['A1'] == plink_results_covs['ALT'])
    plink_results_covs['BETA'] = np.log(plink_results_covs['OR']) 
    plink_results_covs = plink_results_covs.rename(columns={'LOG(OR)_SE':'SE','Z_STAT':'Tstat'})[['#CHROM','POS','ID','REF','ALT','BETA','SE','Tstat','P']].copy()
    plink_results_covs['assoc_test'] = 'LR w/ Covs'

    saige_results = pd.read_csv(saige_filepath)
    saige_results = saige_results.rename(columns={'CHR':'#CHROM','MarkerID':'ID','Allele1':'REF','Allele2':'ALT','p.value':'P'})[['#CHROM','POS','ID','REF','ALT','BETA','SE','Tstat','P']].copy()
    saige_results['assoc_test'] = 'SAIGE w/ Covs'

    # only keep tests in all three (in case plink hit error code)
    common_ids = set(plink_results_nocovs['ID']) \
                & set(plink_results_covs['ID']) \
                & set(saige_results['ID'])
    plink_results_nocovs = plink_results_nocovs[plink_results_nocovs['ID'].isin(common_ids)].copy()
    plink_results_covs   = plink_results_covs[plink_results_covs['ID'].isin(common_ids)].copy()
    saige_results        = saige_results[saige_results['ID'].isin(common_ids)].copy()


    assert plink_results_nocovs.shape[0] == plink_results_covs.shape[0] 
    assert saige_results.shape[0] == plink_results_covs.shape[0]
    assert all(saige_results['ALT'].values==plink_results_nocovs['ALT'].values)
    assert all(saige_results['ALT'].values==plink_results_covs['ALT'].values)

    gwas_merged = pd.concat([plink_results_nocovs,plink_results_covs,saige_results])
    gwas_merged.to_parquet(out)

    os.remove(plink_nocovs_filepath)
    os.remove(plink_covs_filepath)
    os.remove(saige_filepath)

def run_phenotypicsubgroup_gwas(output_dir,output_file_suffix,intermediate_plink_dir,intermediate_saige_dir, phenotypic_subgroup):
    '''
    Runs PLINK and SAIGE GWAS and outputs to parquet file
    '''
    
    # write phenotype file
    phenotype_file_plink=f'{intermediate_plink_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}_{output_file_suffix}'
    phenotype_file_saige=f'{intermediate_saige_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}_{output_file_suffix}'
    generate_phenotype_file(simulation_metadata_path=f'{output_dir}/simulation_metadata_{output_file_suffix}.pkl', phenotypic_subgroup=phenotypic_subgroup, 
                    covar_filepath=f'{output_dir}/COVARIATE_FILE',
                    phenotype_file_plink=phenotype_file_plink,
                    phenotype_file_saige=phenotype_file_saige)

    # run PLINK GWAS (no covariates)
    run_plink_gwas(bfile=f'{output_dir}/G_{output_file_suffix}', covariate_file=None, phenotype_file=phenotype_file_plink, 
                   out=f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_False')

    # run PLINK GWAS (with age and pcs)
    run_plink_gwas(bfile=f'{output_dir}/G_{output_file_suffix}', covariate_file=f'{output_dir}/COVARIATE_FILE', phenotype_file=phenotype_file_plink, 
                   out=f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_True')

    # run SAIGE GWAS (with age and pcs)
    # check both bim df is sorted before running saige gwas
    bim_df =  pd.read_csv(f'{output_dir}/G_{output_file_suffix}.bim',sep='\s+',header=None,names=['CHR','SNP','CM','POS','A1','A2'])
    assert (bim_df['CHR'].diff().fillna(0) >= 0).all(), "CHR not sorted"
    assert all(
        (group['POS'].diff().fillna(0) >= 0).all()
        for _, group in bim_df.groupby('CHR')
    ), "POS not sorted within at least one chromosome"
    run_saige(plinkFile=f'{output_dir}/G',phenoFile=phenotype_file_saige,output_dir=output_dir,
              intermediate_saige_dir=intermediate_saige_dir,
              phenotypic_subgroup=phenotypic_subgroup,output_file_suffix=output_file_suffix)
    
    # consolidate all gwas results into one parquet file
    plink_nocovs_filepath = f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_False.parquet'
    plink_covs_filepath = f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_True.parquet'
    saige_filepath = f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_SAIGE'
    consolidate_files(plink_nocovs_filepath, plink_covs_filepath, saige_filepath, 
                      f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}.parquet')


# evaluate gwas
def evaluate_gwas(output_dir,g_ps,c_ps, e, dataset, phenotypic_subgroup,sig_level=5e-8):
    output_file_suffix = get_output_file_suffix(e=e,g_ps=g_ps,c_ps=c_ps,dataset=dataset)
    meta_path = f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl"
    with open(meta_path, "rb") as f:
        simulation_metadata = pickle.load(f)
    associated_markers = set(simulation_metadata["markers_assoc"][phenotypic_subgroup]) if phenotypic_subgroup in range(2) else set()

    # read parquet with all gwas info
    df = pd.read_parquet(f"{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}.parquet")    
   
    # get confusion matrix
    df = df.assign(is_assoc = df["ID"].isin(associated_markers),is_sig   = df["P"] < sig_level)
    conf_by_method = (
        df.groupby("assoc_test",group_keys=False)
        .apply(lambda g: pd.Series({
            "tp": ( g["is_sig"]  &  g["is_assoc"]).sum(),
            "fp": ( g["is_sig"]  & ~g["is_assoc"]).sum(),
            "tn": (~g["is_sig"]  & ~g["is_assoc"]).sum(),
            "fn": (~g["is_sig"]  &  g["is_assoc"]).sum(),})).reset_index())
    conf_by_method = conf_by_method.assign(
    accuracy  = lambda d: (d.tp + d.tn) / (d.tp + d.fp + d.tn + d.fn),
    precision = lambda d: d.tp / np.where((d.tp + d.fp) == 0, np.nan, d.tp + d.fp),
    recall    = lambda d: d.tp / np.where((d.tp + d.fn) == 0, np.nan, d.tp + d.fn),
    specificity = lambda d: d.tn / np.where((d.tn + d.fp) == 0, np.nan, d.tn + d.fp),
    f1        = lambda d: (2*d.tp) / np.where((2*d.tp + d.fp + d.fn) == 0, np.nan, 2*d.tp + d.fp + d.fn))

    # get change in beta and average abs beta
    results_df = df.groupby('assoc_test',group_keys=False)['BETA'].apply(lambda x: x.abs().mean()).reset_index(name="avg_abs_beta")
    results_df = (results_df.merge(df[df["ID"].isin(associated_markers)].groupby('assoc_test',group_keys=False)['BETA'].apply(lambda x: x.abs().mean())
                                   .reset_index(name="avg_abs_beta_assoc"),on='assoc_test',how='left'))
    wide = df.pivot(index="ID", columns="assoc_test", values="BETA")
    y_true = wide.index.isin(associated_markers)# get those that are truly assoc.
    wide["rel_change_LR_W_COVS"] = ((wide["LR w/ Covs"] - wide["LR"]) / wide["LR"]).abs()
    wide["rel_change_SAIGE_W_COVS"] = ((wide["SAIGE w/ Covs"] - wide["LR"]) / wide["LR"]).abs()
    wide["rel_change_LR_W_COVS_assoc"] = ((wide.loc[y_true]["LR w/ Covs"] - wide.loc[y_true]["LR"]) / wide.loc[y_true]["LR"]).abs()
    wide["rel_change_SAIGE_W_COVS_assoc"] = ((wide.loc[y_true]["SAIGE w/ Covs"] - wide.loc[y_true]["LR"]) / wide.loc[y_true]["LR"]).abs()
    prop_changes = {
        "LR": {'prop_rel_change_gt10':0.0, 'prop_rel_change_gt10_associated':0.0},
        "LR w/ Covs": {'prop_rel_change_gt10':(wide["rel_change_LR_W_COVS"] > 0.10).mean(), 
                    'prop_rel_change_gt10_associated':(wide.loc[wide["rel_change_LR_W_COVS_assoc"].notna(),"rel_change_LR_W_COVS_assoc"] > 0.10).mean()},
        "SAIGE w/ Covs": {'prop_rel_change_gt10':(wide["rel_change_SAIGE_W_COVS"] > 0.10).mean(), 
                        'prop_rel_change_gt10_associated':(wide.loc[wide["rel_change_SAIGE_W_COVS_assoc"].notna(),"rel_change_SAIGE_W_COVS_assoc"] > 0.10).mean()}
    }
    prop_df = pd.DataFrame.from_dict(prop_changes, orient="index").reset_index(names='assoc_test')
    results_df = results_df.merge(prop_df,on='assoc_test')
    results_df = results_df.merge(conf_by_method,on='assoc_test')

    # set param values
    results_df['g_ps'] = g_ps
    results_df['c_ps'] = c_ps
    results_df['e'] = e
    results_df['dataset'] = dataset
    results_df['phenotypic_subgroup'] = phenotypic_subgroup

    return results_df