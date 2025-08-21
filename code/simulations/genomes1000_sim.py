import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os

def prep_1000genomes_bed_file(root_dir, intermediate_file_dir):
    # change map and ped files to bed format
    # major allele set to A2 (If a binary fileset was originally loaded, --keep-allele-order forces the original A1/A2 allele encoding to be preserved; otherwise, the major allele is set to A2)
    plink_extract = f'''
    module load plink/1.9 && plink --file {root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05 \
        --make-bed \
        --out {intermediate_file_dir}/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

def sun_generate_sim_data(root_dir,intermediate_file_dir,intermediate_file_suffix,ps,num_markers_assoc,e,extra_subgroups_size,M,num_clinical_assoc, K=5):
    '''
    Generate synthetic data similar to Sun et al. (Multi-view biclustering for genotype-phenotype association studies of complex diseases)
    using 1000 Genomes Phase 3 data. Use admixture files which contain 193634 markers with MAF>5% and 2504 individuals. 

    PARAMS:
    root_dir: root directory for 1000 Genomes data
    intermediate_file_dir: dir to write intermediate files to (when using plink for example)
    intermediate_file_suffix: such that if multiple simulations are created, each is distinctly defined
    M: number of clinical features (right now assuming all from one domain & all binary)
    ps: variable controlling how much population stratification is affecting geno-pheno relationship (needs to be in range(0,1,size=0.1)) 
    (0-> pick SNPs in bottom 10% by allele frequency variance i.e. little pop. strat., 0.9-> pick SNPS in top 10% by allele frequency variance i.e. large pop. strat.)
    num_markers_assoc: number of markers with an associated with subtype classification (if rij>int(0.4*markers_assoc) then subject i in subgroup j)
    e: relative effect that genetic variation contributed to the effect of the phenotype. e in [0,1]. (decreased e means higher level of disagreement between genotypic and phenotypic subgroups)
    num_clinical_assoc: number of clinical features associated with subtype classification (same for all subtypes)
    extra_subgroups_size: number of people in s3 and s4 (selected at random)
    K: number of admixture groups to estimate af variance over (K in [5,26] per 1000 genomes phase 3 paper)

    outputs:
    genetic_subgroups: genetic subgroup assignments for all individuals
    phenotypic_subgroups: phenotypic subgroup assignments for all individuals
    C: clinical data matrix (num samples x M)
    iid_order: IIDs in order (to align to clinical data matrix)
    markers_assoc_dict: for each genetic subgroup, IDs of genetic markers selected to be associated
    clinical_assoc_df: for each phenotypic subgroup, index of clinical variables selected to be associated (and their selected assoc. strength)

    '''
    assert num_clinical_assoc<M, "num_clinical_assoc cannot exceed M"

    # 1. Read in allele frequencies per 5 admixture groups to estimate af variance across groups
    admixture_af = pd.read_csv(f"{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.{K}.P", sep='\s+',header=None,names=range(K))
    map = pd.read_csv(f"{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map", sep='\s+', header=None,
                        names=["CHR","ID","GEN_DIST","BP"])
    # add SNP information to admixture af df
    admixture_af = pd.concat([map,admixture_af],axis=1)
    admixture_af['af_variance'] = admixture_af[range(K)].var(axis=1)
    admixture_af['af_var_decile'] = (pd.qcut(admixture_af['af_variance'], 10, labels=False))/10 # discretize into equal size buckets based on deciles

    # 2. Generate genetic subgroups
    genetic_subgroups = []
    markers_assoc_dict = {} # names of the markers that are associated with each subgroup
    for genetic_subgroup in range(2):
        # Select SNP group based on num_markers_assoc and ps
        assert admixture_af[admixture_af['af_var_decile']==ps].shape[0]>num_markers_assoc, f"number of genetic features assoc. ({num_markers_assoc}) is too large, only {admixture_af[admixture_af['af_var_decile']==ps].shape[0]} markers in decile {ps} group"
        markers_assoc = admixture_af[admixture_af['af_var_decile']==ps].sample(n=num_markers_assoc, replace=False)['ID'].values.tolist()
        markers_assoc_dict[genetic_subgroup] = markers_assoc
        assert len(set(markers_assoc))==len(markers_assoc) # make sure ped file has unique rows

        
        # extract selected markers
        with open(f'{intermediate_file_dir}/markers_assoc_g{genetic_subgroup}_{intermediate_file_suffix}.txt','w') as f:
            for snp in markers_assoc:
                f.write(snp + "\n")

        if not os.path.exists(f'{intermediate_file_dir}/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.bed'):
            prep_1000genomes_bed_file(root_dir=root_dir, intermediate_file_dir=intermediate_file_dir)
        
        # extract select markers and get marker values for each individual (0 - no copies of minor allele, 1 - 1 copy of minor allele, 2 - 2 copies of minor allele)
        plink_extract = f'''
        module load plink/1.9 && plink --bfile {intermediate_file_dir}/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05 \
            --extract {intermediate_file_dir}/markers_assoc_g{genetic_subgroup}_{intermediate_file_suffix}.txt \
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
        assert len(geno_cols) == num_markers_assoc
        raw[geno_cols] = (raw[geno_cols] > 0).astype(int) # recode s.t. values 1 and 2 map to 1
        genetic_subgroup_df = raw.set_index('IID')[geno_cols].sum(axis=1).reset_index(name=f'r')
        genetic_subgroup_df['genetic_subgroup'] = genetic_subgroup
        
        deciles, bins = pd.qcut(genetic_subgroup_df["r"], 10, labels=False, retbins=True)
        genetic_subgroup_df[f'subgroup'] =genetic_subgroup_df['r']>bins[-3] # Top 20% of people per r
        genetic_subgroups.append(genetic_subgroup_df)
    genetic_subgroups = pd.concat(genetic_subgroups)

    # 3. Generate phenotypic subgroups
    iid_order = raw.IID.values # can use raw file from whichever subgroup, since IID always in same order
    # can just use bins[-4] - somewhat equivalent to 7.5
    phenotypic_subgroups = []
    for phenotypic_subgroup in range(2):
        phenotypic_subgroup_df = genetic_subgroups[genetic_subgroups['genetic_subgroup']==phenotypic_subgroup][['IID','r']].copy()
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        phenotypic_subgroup_df['subgroup'] = phenotypic_subgroup_df['r']*e + np.random.randn(len(phenotypic_subgroup_df)) > bins[-4]*e
        phenotypic_subgroups.append(phenotypic_subgroup_df)
    for phenotypic_subgroup in range(2,4):
        # randomly select extra_subgroups_size people
        randomly_selected = pd.Series(iid_order).sample(extra_subgroups_size).values.tolist() 
        phenotypic_subgroup_df = pd.DataFrame(iid_order,columns=['IID'])
        phenotypic_subgroup_df['r'] = None
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        phenotypic_subgroup_df['subgroup'] = phenotypic_subgroup_df['IID'].isin(randomly_selected)
        phenotypic_subgroups.append(phenotypic_subgroup_df)
    phenotypic_subgroups = pd.concat(phenotypic_subgroups)

    # 4. simulate M binary clinical features
    # start with baseline probabiliyies
    probs = np.full((len(iid_order), M), 0.1, dtype=float) # baseline prob of clinical feature is 0.1
    clinical_assoc_df_rows = [] # index of clinical vars that are associated (and their strength)
    for phenotypic_subgroup in range(4):
        # index of randomly chosen, associated clinical variables 
        assoc_idx = np.random.choice(M, size=num_clinical_assoc, replace=False) 
        np.random.shuffle(assoc_idx)       
        n1 = num_clinical_assoc // 3 # 1/3 who get P 0.6
        n2 = 2 * num_clinical_assoc // 3 # 1/3 who get P 0.5
        # get those who are in the subgroup
        mask = ((phenotypic_subgroups["phenotypic_subgroup"] == phenotypic_subgroup) & (phenotypic_subgroups["subgroup"]))
        subj_ids = phenotypic_subgroups.loc[mask, "IID"].unique()
        # map subject IDs to row indices # TODO: pick up from here
        row_idx = [i for i, iid in enumerate(iid_order) if iid in subj_ids]
        probs[np.ix_(row_idx, assoc_idx[:n1])] = 0.6
        probs[np.ix_(row_idx, assoc_idx[n1:n2])] = 0.5
        probs[np.ix_(row_idx, assoc_idx[n2:])] = 0.4
        clinical_assoc_df_rows += [
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.6, "indices": assoc_idx[:n1]},
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.5, "indices": assoc_idx[n1:n2]},
        {"phenotypic_subgroup": phenotypic_subgroup, "strength": 0.4, "indices": assoc_idx[n2:]},
        ]
    C = (np.random.rand(len(iid_order), M) < probs).astype(int)
    clinical_assoc_df = pd.DataFrame(clinical_assoc_df_rows)


    return genetic_subgroups, phenotypic_subgroups, C, iid_order, markers_assoc_dict, clinical_assoc_df # TODO: need to get markers assoc and clinical assoc