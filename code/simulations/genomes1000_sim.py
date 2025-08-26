import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
import pickle
from sklearn.preprocessing import StandardScaler

def prep_1000genomes_bed_file(root_dir, output):
    # change map and ped files to bed format
    # major allele set to A2 (If a binary fileset was originally loaded, --keep-allele-order forces the original A1/A2 allele encoding to be preserved; otherwise, the major allele is set to A2)
    plink_extract = f'''
    module load plink/1.9 && plink --file {root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05 \
        --make-bed \
        --out {output}
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

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

def sun_generate_sim_data(bfile_path, maf_by_superpop_filepath,
                          intermediate_file_dir,intermediate_file_suffix,output_dir,output_file_suffix,
                          ps,num_markers_assoc,e,extra_subgroups_size,M,num_clinical_assoc):
    '''
    Generate synthetic data similar to Sun et al. (Multi-view biclustering for genotype-phenotype association studies of complex diseases)
    using 1000 Genomes Phase 3 data. Use admixture files which contain 193634 markers with MAF>5% and 2504 individuals. 

    PARAMS:
    bfile_path: path to bfile for genetic data
    maf_by_superpop_filepath: plink generated .frq.strat file for maf within each superpopulation group. don't include .frq.strat suffix in filename.
    intermediate_file_dir: dir to write intermediate files to (when using plink for example)
    intermediate_file_suffix: such that if multiple simulations are created, each is distinctly defined
    output_file_suffix: such that if multiple simulations are created, each is distinctly defined - suffic for C and simulated data pkl file
    output_dir: where to write output genetic data matrix (X) in form of plink bfile, and clinical data matrix C
    M: number of clinical features (right now assuming all from one domain & all binary)
    ps: variable controlling how much population stratification is affecting geno-pheno relationship (needs to be in range(0,1,size=0.1)) 
    (0-> pick SNPs in bottom 10% by allele frequency variance i.e. little pop. strat., 0.9-> pick SNPS in top 10% by allele frequency variance i.e. large pop. strat.)
    num_markers_assoc: number of markers with an associated with subtype classification (if rij>int(0.4*markers_assoc) then subject i in subgroup j)
    e: relative effect that genetic variation contributed to the effect of the phenotype. e in [0,1]. (decreased e means higher level of disagreement between genotypic and phenotypic subgroups)
    num_clinical_assoc: number of clinical features associated with subtype classification (same for all subtypes)
    extra_subgroups_size: number of people in s3 and s4 (selected at random)

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

    # 1. Read in allele frequencies per 5 superpopulations to estimate af variance across groups
    maf_by_superpop = pd.read_csv(f'{maf_by_superpop_filepath}.frq.strat',sep='\s+')
    superpopulations = maf_by_superpop['CLST'].unique()
    maf_by_superpop = maf_by_superpop.pivot(index=['SNP'],columns='CLST',values='MAF').reset_index()
    assert maf_by_superpop.shape[0] == 193634
    maf_by_superpop['af_variance'] = maf_by_superpop[superpopulations].var(axis=1)
    maf_by_superpop['af_var_decile'] = (pd.qcut(maf_by_superpop['af_variance'], 10, labels=False))/10 # discretize into equal size buckets based on deciles


    # 2. Generate genetic subgroups
    genetic_subgroups = []
    markers_assoc_dict = {} # names of the markers that are associated with each subgroup
    for genetic_subgroup in range(2):
        # Select SNP group based on num_markers_assoc and ps
        assert maf_by_superpop[maf_by_superpop['af_var_decile']==ps].shape[0]>num_markers_assoc, f"number of genetic features assoc. ({num_markers_assoc}) is too large, only {maf_by_superpop[maf_by_superpop['af_var_decile']==ps].shape[0]} markers in decile {ps} group"
        markers_assoc = maf_by_superpop[maf_by_superpop['af_var_decile']==ps].sample(n=num_markers_assoc, replace=False)['SNP'].values.tolist()
        markers_assoc_dict[genetic_subgroup] = markers_assoc
        assert len(set(markers_assoc))==len(markers_assoc) # make sure ped file has unique rows

        
        # extract selected markers
        with open(f'{intermediate_file_dir}/markers_assoc_g{genetic_subgroup}_{intermediate_file_suffix}.txt','w') as f:
            for snp in markers_assoc:
                f.write(snp + "\n")
        
        # extract select markers and get marker values for each individual (0 - no copies of minor allele, 1 - 1 copy of minor allele, 2 - 2 copies of minor allele)
        plink_extract = f'''
        module load plink/1.9 && plink --bfile {bfile_path} \
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
    fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = fam_df['IID'].values
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
    C = np.random.poisson(0.1,(len(iid_order), M)) # baseline prob of clinical feature is 0.1
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
        # map subject IDs to row indices 
        row_idx = [i for i, iid in enumerate(iid_order) if iid in subj_ids]
        C[np.ix_(row_idx, assoc_idx[:n1])] = np.random.poisson(1, size=(len(row_idx), n1))
        C[np.ix_(row_idx, assoc_idx[n1:n2])] = np.random.poisson(0.75, size=(len(row_idx), n2-n1))
        C[np.ix_(row_idx, assoc_idx[n2:])] = np.random.poisson(0.5, size=(len(row_idx), len(assoc_idx)-n2))
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
    "clinical_assoc": clinical_assoc_df              # pandas DataFrame
    }
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return genetic_subgroups, phenotypic_subgroups, C, iid_order, markers_assoc_dict, clinical_assoc_df 
    



# Functions for data simulation evaluation
def clean_join(values):
    # remove empty strings
    vals = sorted(set(v for v in values if v != ''), key=lambda x: int(x))
    return ','.join(vals)

def generate_umap_plot(mode, var_list, color_col, color_label, output_dir, igsr_samples_filepath=None):
    '''Generate UMAP plot of C across different values of variable spcified in 'mode' 
    (values in var_list), colored by color_df (which must have a column IID)'''
    plot_dfs = []
    for var in var_list:
        if mode == 'ps':
            output_suffix = f'ps_{var}_e_0.5'
            with open(f"{output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            color_df = read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/X')
        else:
            assert mode=='e', "only works with modes ps and e so far"
            output_suffix = f'ps_0.5_e_{var}'
            with open(f"{output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            genetic_subgroups = simulation_metadata['genetic_subgroups']
            genetic_subgroups['subgroup_value'] = np.where(genetic_subgroups['subgroup'],genetic_subgroups['genetic_subgroup'],'') # change this to one label per person
            genetic_subgroups_concat = genetic_subgroups.groupby('IID')['subgroup_value'].apply(clean_join).reset_index()
            genetic_subgroups_concat["IID"] = pd.Categorical(genetic_subgroups_concat["IID"], categories=iid_order, ordered=True)
            color_df = genetic_subgroups_concat.copy()
        C = np.load(f'{output_dir}/C_{output_suffix}.npy')# pick e=0.5
        C_scaled = StandardScaler().fit_transform(C) # standardize matrix before UMAP

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(C_scaled)

        plot_df = (pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=iid_order).
                    rename_axis("IID").reset_index().merge(color_df[['IID',color_col]], on='IID',how='inner'))
        
        assert plot_df[plot_df[color_col].isna()].shape[0] == 0
        plot_df[mode] = var
        plot_dfs.append(plot_df)
    plot_dfs = pd.concat(plot_dfs)

    # --- Plot with plotnine ---
    if mode == 'ps':
        title = r"UMAP of clinical data at different $p_s$ levels"
    else:
        title = r"UMAP of clinical data at different e levels"
    p = (
        ggplot(plot_dfs, aes("UMAP1", "UMAP2", color=color_col))
        + geom_point(alpha=0.5, size=2)
        + labs(title=title, color=color_label)
        + facet_wrap(f'~{mode}',ncol=2,scales='free')
        + theme_minimal()
        + theme(figure_size=(8, 12),legend_title=element_text(size=9))
    )
    p.save(f'{output_dir}/umap_clinical_{mode}.pdf',dpi=300)

def run_phenotypicsubgroup_gwas(output_dir,intermediate_file_dir,ps,e,cov_included,phenotypic_subgroup):
    # define specific file paths
    cov_file_suffix = '' if cov_included else '_NOPS'
    output_suffix = f'ps_{ps}_e_{e}'

    # write phenotype file
    with open(f"{output_dir}/simulation_metadata_{output_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    phenotypic_subgroups = simulation_metadata['phenotypic_subgroups']
    pheno = phenotypic_subgroups[phenotypic_subgroups['phenotypic_subgroup']==phenotypic_subgroup][['IID','subgroup']].rename(columns={'subgroup':'Phenotype'})
    pheno['FID'] = pheno['IID']
    pheno['Phenotype'] = pheno['Phenotype'].astype(int)
    pheno[['FID','IID','Phenotype']].set_index('FID').to_csv(f'{intermediate_file_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}')

    # run GWAS
    result = subprocess.run(f'module unload plink && module load plink/2.0a5.13 && plink --bfile {output_dir}/X\
                        --covar {intermediate_file_dir}/COVARIATE_FILE{cov_file_suffix} --covar-variance-standardize\
                        --pheno {intermediate_file_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}\
                        --glm omit-ref\
                        --out {output_dir}/GWAS_RESULTS/PhenotypicSubgroup_{phenotypic_subgroup}_Geno_Cov_{cov_included}_ps_{ps}_e_{e}\
                        --1 --no-pheno', shell=True, capture_output=True, text=True, executable='/bin/bash')
    result.check_returncode()