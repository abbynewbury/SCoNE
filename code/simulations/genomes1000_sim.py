import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
import glob
import pickle
from sklearn.preprocessing import StandardScaler

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
        {"--thin-count 10000 --seed 42" if subset_test else ""}\
        --make-bed \
        --out {output}
    '''
    result = subprocess.run(plink_extract, shell=True, check=True, executable="/bin/bash")

    # make .raw file for G matrix later
    plink_extract = f'''
    module load plink/1.9 && plink --bfile {output} \
        --recode A \
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


def get_output_file_suffix(ps,e,init,num_markers_assoc):
    return f'ps_{ps}_e_{e}_init_{init}_markersassoc_{num_markers_assoc}'

def rs(streams,stream_name):
        return int(streams[stream_name].integers(1, 2**31 - 1))

def sun_generate_sim_data(bfile_path, af_df_filepath,map_filepath,
                          intermediate_file_dir,intermediate_file_suffix,output_dir,output_file_suffix,
                          ps,num_markers_assoc,e,extra_subgroups_size,M,num_clinical_assoc, run_seed):
    '''
    Generate synthetic data similar to Sun et al. (Multi-view biclustering for genotype-phenotype association studies of complex diseases)
    using 1000 Genomes Phase 3 data. Use admixture files which contain 193634 markers with MAF>5% and 2504 individuals. 

    PARAMS:
    bfile_path: path to bfile for genetic data
    af_df_filepath: pre-generated admixture fractions for K=5 (release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5)
    map_filepath: map file corresponding to admixture fraction file creation (release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map)
    admixture_fractions_filepath: path
    intermediate_file_dir: dir to write intermediate files to (when using plink for example)
    intermediate_file_suffix: such that if multiple simulations are created, each is distinctly defined
    output_file_suffix: such that if multiple simulations are created, each is distinctly defined - suffix for C and simulated data pkl file
    output_dir: where to write output genetic data matrix (X) in form of plink bfile, and clinical data matrix C
    M: number of clinical features (right now assuming all from one domain & all binary)
    ps: variable controlling how much population stratification is affecting geno-pheno relationship (needs to be in range(0,1,size=0.1)) 
    (0-> pick SNPs in bottom 10% by allele frequency variance i.e. little pop. strat., 0.9-> pick SNPS in top 10% by allele frequency variance i.e. large pop. strat.)
    num_markers_assoc: number of markers with an associated with subtype classification (if rij>int(0.4*markers_assoc) then subject i in subgroup j)
    e: relative effect that genetic variation contributed to the effect of the phenotype. e in [0,1]. (decreased e means higher level of disagreement between genotypic and phenotypic subgroups)
    num_clinical_assoc: number of clinical features associated with subtype classification (same for all subtypes)
    extra_subgroups_size: number of people in s3 and s4 (selected at random)
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
    names = ["markers", "noise", "extra_sub", "assoc", "poisson"]
    streams = {name: np.random.default_rng(ss) for name, ss in zip(names, parent_ss.spawn(len(names)))}

    # 1. Read in allele frequencies per 5 admixture fractions to estimate af variance across groups
    assert ps in [True,False], f"{ps} is not a valid value for ps, must be in [True,False]" 
    af_variance_df = pd.read_csv(f'{af_df_filepath}.P',header=None,sep='\s+')
    # get marker id
    df_map = pd.read_csv(map_filepath,sep='\s+',header=None,names=["chrom", "SNP", "cm", "bp"])
    af_variance_df['SNP'] = df_map['SNP'].values
    # make sure af variance df and bim df have same snps (incase of subsetting)
    bim_df = pd.read_csv(f'{bfile_path}.bim',sep='\s+',header=None,names=['CHR','SNP','CM','POS','A1','A2'])
    af_variance_df = af_variance_df.merge(bim_df[['SNP']], on='SNP', how='inner')
    admixture_indiv_df = pd.read_csv(f'{af_df_filepath}.Q',header=None,sep='\s+')

    X = af_variance_df.iloc[:, 0:5].to_numpy(float)      # N x 5
    w = admixture_indiv_df.mean(axis=0).to_numpy(float)  # length 5
    w = w / w.sum()
    mu = X @ w                                           # (N,)
    af_variance_df['af_variance_weighted'] = ((X - mu[:, None])**2 * w[None, :]).sum(axis=1)
    af_variance_df['af_var_quartile'] = (pd.qcut(af_variance_df['af_variance_weighted'], 4, labels=[0.00, 0.25, 0.50, 0.75]))
    af_variance_df['ps'] = af_variance_df['af_var_quartile'].map({0.00: False, 0.75: True})

    # 2. Generate genetic subgroups
    genetic_subgroups = []
    markers_assoc_dict = {} # names of the markers that are associated with each subgroup
    for genetic_subgroup in range(2):
        # Select SNP group based on num_markers_assoc and ps 
        assert af_variance_df[af_variance_df['ps']==ps].shape[0]>num_markers_assoc, f"number of genetic features assoc. ({num_markers_assoc}) is too large, only {af_variance_df[af_variance_df['ps']==ps].shape[0]} markers in quartile {ps} group"
        markers_assoc = af_variance_df[af_variance_df['ps']==ps].sample(n=num_markers_assoc, replace=False, random_state=rs(streams,"markers"))['SNP'].values.tolist()
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
        genetic_subgroup_df[f'subgroup'] =genetic_subgroup_df['r']>genetic_subgroup_df['r'].quantile(0.8) # Top 20% of people per r
        genetic_subgroups.append(genetic_subgroup_df)
    genetic_subgroups = pd.concat(genetic_subgroups)

    # 3. Generate phenotypic subgroups
    fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = fam_df['IID'].values
    phenotypic_subgroups = []
    for phenotypic_subgroup in range(2): 
        phenotypic_subgroup_df = genetic_subgroups[genetic_subgroups['genetic_subgroup']==phenotypic_subgroup][['IID','r']].copy()
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        # corresponding genetic subgroup value for r
        phenotypic_subgroup_df['subgroup'] = phenotypic_subgroup_df['r']*e + streams["noise"].normal(loc=0,scale=0.1*phenotypic_subgroup_df['r'].std(),size=len(phenotypic_subgroup_df)) > (phenotypic_subgroup_df['r'].quantile(0.8))*e
        phenotypic_subgroups.append(phenotypic_subgroup_df)
    for phenotypic_subgroup in range(2,4): 
        # randomly select extra_subgroups_size people
        randomly_selected = pd.Series(iid_order).sample(extra_subgroups_size,random_state=rs(streams,"extra_sub")).values.tolist() 
        phenotypic_subgroup_df = pd.DataFrame(iid_order,columns=['IID'])
        phenotypic_subgroup_df['r'] = None
        phenotypic_subgroup_df['phenotypic_subgroup'] = phenotypic_subgroup
        phenotypic_subgroup_df['subgroup'] = phenotypic_subgroup_df['IID'].isin(randomly_selected)
        phenotypic_subgroups.append(phenotypic_subgroup_df)
    phenotypic_subgroups = pd.concat(phenotypic_subgroups)

    # 4. simulate M binary clinical features
    # start with baseline probabiliyies
    C = streams["poisson"].poisson(0.1,(len(iid_order), M)) # baseline prob of clinical feature is 0.1 
    clinical_assoc_df_rows = [] # index of clinical vars that are associated (and their strength)
    for phenotypic_subgroup in range(4):
        # index of randomly chosen, associated clinical variables 
        assoc_idx = streams["assoc"].choice(M, size=num_clinical_assoc, replace=False)  
        n1 = num_clinical_assoc // 3 # 1/3 who get P 0.6
        n2 = 2 * num_clinical_assoc // 3 # 1/3 who get P 0.5
        # get those who are in the subgroup
        mask = ((phenotypic_subgroups["phenotypic_subgroup"] == phenotypic_subgroup) & (phenotypic_subgroups["subgroup"]))
        subj_ids = phenotypic_subgroups.loc[mask, "IID"].unique()
        # map subject IDs to row indices 
        row_idx = [i for i, iid in enumerate(iid_order) if iid in subj_ids]
        C[np.ix_(row_idx, assoc_idx[:n1])] = streams["poisson"].poisson(1, size=(len(row_idx), n1))
        C[np.ix_(row_idx, assoc_idx[n1:n2])] = streams["poisson"].poisson(0.75, size=(len(row_idx), n2-n1))
        C[np.ix_(row_idx, assoc_idx[n2:])] = streams["poisson"].poisson(0.5, size=(len(row_idx), len(assoc_idx)-n2))
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

# UMAP RELATED
def clean_join(values):
    # remove empty strings
    vals = sorted(set(v for v in values if v != ''), key=lambda x: int(x))
    return ','.join(vals)

def generate_umap_plot(mode, var_list, color_col, color_label, output_dir, igsr_samples_filepath=None): 
    '''Generate UMAP plot of C across different values of variable spcified in 'mode' 
    (values in var_list), colored by color_df (which must have a column IID)
    
    output dir and output suffix define where to find simulated data files
    '''
    plot_dfs = []
    for var in var_list:
        if mode == 'ps':
            output_file_suffix = get_output_file_suffix(ps=var,e=0.5,init=0,num_markers_assoc=2000) # choose first initialization and 2000 markers assoc for vis purposes
            with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            color_df = read_in_igsr_samples(igsr_samples_filepath, bfile_path=f'{output_dir}/G')
        else:
            assert mode=='e', "only works with modes ps and e so far"
            output_file_suffix = get_output_file_suffix(ps=False,e=var,init=0,num_markers_assoc=2000) # choose first initialization and 2000 markers assoc for vis purposes
            with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
                simulation_metadata = pickle.load(f)
            iid_order = simulation_metadata['iid_order']
            genetic_subgroups = simulation_metadata['genetic_subgroups']
            genetic_subgroups['subgroup_value'] = np.where(genetic_subgroups['subgroup'],genetic_subgroups['genetic_subgroup'],'') # change this to one label per person
            genetic_subgroups_concat = genetic_subgroups.groupby('IID')['subgroup_value'].apply(clean_join).reset_index()
            genetic_subgroups_concat["IID"] = pd.Categorical(genetic_subgroups_concat["IID"], categories=iid_order, ordered=True)
            color_df = genetic_subgroups_concat.copy()
        C = np.load(f'{output_dir}/C_{output_file_suffix}.npy')# pick e=0.5
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
        + theme(legend_title=element_text(size=9))
    )
    p.save(f'{output_dir}/umap_clinical_{mode}.pdf',dpi=300)

# GWAS RELATED

def run_phenotypicsubgroup_gwas(output_dir,output_file_suffix,intermediate_file_dir,cov_included,phenotypic_subgroup):
    '''
    Runs PLINK and SAIGE GWAS and outputs to parquet file
    '''
    # define specific file paths
    cov_file_suffix = '' if cov_included else '_NOPS'

    # write phenotype file
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
        simulation_metadata = pickle.load(f)
    phenotypic_subgroups = simulation_metadata['phenotypic_subgroups']
    pheno = phenotypic_subgroups[phenotypic_subgroups['phenotypic_subgroup']==phenotypic_subgroup][['IID','subgroup']].rename(columns={'subgroup':'Phenotype'})
    pheno['FID'] = pheno['IID']
    pheno['Phenotype'] = pheno['Phenotype'].astype(int)
    pheno[['FID','IID','Phenotype']].set_index('FID').to_csv(f'{intermediate_file_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}_{output_file_suffix}')

    # run GWAS
    result = subprocess.run(f'module unload plink && module load plink/2.0a5.13 && plink --bfile {output_dir}/G\
                        --covar {intermediate_file_dir}/COVARIATE_FILE{cov_file_suffix} --covar-variance-standardize\
                        --pheno {intermediate_file_dir}/PHENOTYPE_FILE_Subgroup{phenotypic_subgroup}_{output_file_suffix}\
                        --glm omit-ref\
                        --out {output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}\
                        --1 --no-pheno', shell=True, capture_output=True, text=True, executable='/bin/bash')
    result.check_returncode()

    # write plink results to parquet for quicker analysis
    with open(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}.log','r') as f:
        file = f.read()
        assert "End time" in file, f"plink ended with errors for {output_file_suffix}"
    plink_results = pd.read_csv(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}.Phenotype.glm.logistic.hybrid',sep='\t')
    plink_results = plink_results[plink_results['TEST']=='ADD'].copy() # only write SNP effect size data
    plink_results["OR"] = pd.to_numeric(plink_results["OR"])
    plink_results.to_parquet(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}_results.parquet', engine='pyarrow') # export to parquet format for quicker lookup later on
    # clean up for storage space
    os.remove(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}.log')
    os.remove(f'{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_{cov_included}.Phenotype.glm.logistic.hybrid')

# gwas evaluation

def counts_from_bool(true_arr, pred_arr):
    tp = np.count_nonzero(pred_arr & true_arr)
    tn = np.count_nonzero(~pred_arr & ~true_arr)
    fp = np.count_nonzero(pred_arr & ~true_arr)
    fn = np.count_nonzero(~pred_arr & true_arr)
    return tn, fp, fn, tp

def metrics_from_counts(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    acc  = (tp + tn) / n 
    prec = tp / (tp + fp) if (tp+fp)>0 else np.nan
    rec  = tp / (tp + fn) if (tp+fn)>0 else np.nan
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec)>0 else np.nan
    spec = tn / (tn + fp) if (tn + fp)>0 else np.nan
    return acc, prec, rec, f1, spec

def evaluate_gwas(output_dir,ps, e, init, num_markers_assoc, phenotypic_subgroup,sig_level=5e-8):
    output_file_suffix = sim_functions.get_output_file_suffix(ps, e, init, num_markers_assoc)
    meta_path = f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl"

    with open(meta_path, "rb") as f:
        simulation_metadata = pickle.load(f)


    # read both covariate and no-covariate files once
    df_with = (pd.read_parquet(
        f"{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_True_results.parquet")
        .rename(columns={"OR": "OR_with_pcs", "P": "P_with_pcs","LOG(OR)_SE":"LOG(OR)_SE_with_pcs"}))
    
    df_with = df_with[(df_with['ERRCODE']=='.')].copy() 

    df_nopcs = (pd.read_parquet(
        f"{output_dir}/GWAS_RESULTS/PhenotypicSubgroup{phenotypic_subgroup}_{output_file_suffix}_Geno_Cov_False_results.parquet")
        .rename(columns={"OR": "OR_no_pcs", "P": "P_no_pcs","LOG(OR)_SE":"LOG(OR)_SE_no_pcs"}))
    df_nopcs = df_nopcs[(df_nopcs['ERRCODE']=='.')].copy() # badly behaved SNP, can remove with HWE filtering (check though)

    wide = df_nopcs.merge(df_with, on="ID", how="inner")
    
    associated_markers = set(simulation_metadata["markers_assoc"][phenotypic_subgroup]) if phenotypic_subgroup in range(2) else set()
    y_true = wide["ID"].isin(associated_markers).to_numpy() # true associations

    yhat_no = (wide["P_no_pcs"].to_numpy() < sig_level) # predicted associations no PCs
    yhat_with = (wide["P_with_pcs"].to_numpy() < sig_level) # predicted associations with PCs

    tn0, fp0, fn0, tp0 = counts_from_bool(y_true, yhat_no) # tn, ... without PCs
    tn1, fp1, fn1, tp1 = counts_from_bool(y_true, yhat_with) # tn,... with PCs

    acc0, prec0, rec0, f10, spec0 = metrics_from_counts(tn0, fp0, fn0, tp0) # without PCs
    acc1, prec1, rec1, f11, spec1 = metrics_from_counts(tn1, fp1, fn1, tp1) # with PCs

    
    rel_change = (wide["OR_with_pcs"] - wide["OR_no_pcs"]) / wide["OR_no_pcs"]
    prop_changed = (rel_change.abs() > 0.10).mean() # proportion where OR changes +- 10% in presence of pcs
    rel_change_associated = (wide.loc[y_true]["OR_with_pcs"] - wide.loc[y_true]["OR_no_pcs"]) / wide.loc[y_true]["OR_no_pcs"]
    prop_changed_associated = (rel_change_associated.abs() > 0.10).mean() # proportion where OR changes +- 10% in presence of pcs
    avg_or_associated_with_pcs = (wide.loc[y_true]["OR_with_pcs"]).mean()
    avg_or_associated_no_pcs = (wide.loc[y_true]["OR_no_pcs"]).mean()
    avg_se_associated_with_pcs = (wide.loc[y_true]["LOG(OR)_SE_with_pcs"]).mean()
    avg_se_associated_no_pcs = (wide.loc[y_true]["LOG(OR)_SE_no_pcs"]).mean()

    return dict(
        ps=ps, e=e, init=init, num_markers_assoc=num_markers_assoc, phenotypic_subgroup=phenotypic_subgroup,
        tn_no_pcs=tn0, fp_no_pcs=fp0, fn_no_pcs=fn0, tp_no_pcs=tp0,
        tn_with_pcs=tn1, fp_with_pcs=fp1, fn_with_pcs=fn1, tp_with_pcs=tp1,
        acc_no_pcs=acc0,  prec_no_pcs=prec0,  rec_no_pcs=rec0,  f1_no_pcs=f10, spec_no_pcs=spec0,
        acc_with_pcs=acc1, prec_with_pcs=prec1, rec_with_pcs=rec1, f1_with_pcs=f11, spec_with_pcs=spec1,
        prop_changed=prop_changed, prop_changed_associated=prop_changed_associated, 
        avg_or_associated_with_pcs=avg_or_associated_with_pcs, avg_or_associated_no_pcs=avg_or_associated_no_pcs,
        avg_se_associated_with_pcs=avg_se_associated_with_pcs,avg_se_associated_no_pcs=avg_se_associated_no_pcs
    )