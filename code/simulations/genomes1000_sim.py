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
from cyvcf2 import VCF
from multiprocessing import Pool


def get_variant_exon_df(root_dir,output_csv):
    cmd = f"""
    module load bcftools/1.21 && module load parallel && \
    export BCFTOOLS_PLUGINS=$(bcftools +plugins 2>/dev/null | awk '/Directory/ {{print $2}}'); \
    parallel -j6 '
    bcftools view -i "ID ~ \\"^rs\\"" -Ou {{}} |
    bcftools +split-vep -a CSQ -d -f "%ID\\t%Allele\\t%SYMBOL\\t%Consequence\\n" -
    ' ::: {root_dir}/release-20130502-supporting/functional_annotation/unfiltered/ALL.chr{{1..22}}.phase3_shapeit2_mvncall_integrated_v5_func_anno.20130502.sites.vcf.gz \
    > "{output_csv}"
    """
    result = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    variant_gene = pd.read_csv(output_csv,sep='\t',header=None,names=["variant","allele","symbol","consequence"])
    # only keep variants that are coding
    variant_gene = variant_gene[variant_gene['symbol']!='.'].copy()
    exon_terms = ['missense_variant','synonymous_variant','stop_gained','stop_lost','start_lost','frameshift_variant','coding_sequence_variant']
    variant_gene = variant_gene[variant_gene['consequence'].isin(exon_terms)].copy()
    # avoid duplicate variant-gene pairs
    variant_gene = variant_gene.drop_duplicates(['variant','allele']).copy()
    # only keep those genes with variant count above 5    
    vc = variant_gene['symbol'].value_counts()
    keep_genes = vc[vc > 5].index
    variant_gene = variant_gene[variant_gene['symbol'].isin(keep_genes)].copy()
    # keep those where length of allele = 1
    variant_gene = variant_gene[variant_gene['allele'].str.len() == 1].copy()

    variant_gene.to_csv(output_csv,sep='\t')
    return variant_gene

def run_plink(chr,root_dir,intermediate_dir):
    cmd = f'''module load plink/1.9 && plink --vcf {root_dir}/GRCh37/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz\
            --extract {intermediate_dir}/ids.txt\
            --biallelic-only strict --maf 0.05 --geno 0.05 --mind 0.05 --make-bed --out {intermediate_dir}/chr{chr}.preprune'''
    result = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash") 

def prep_1000genomes_bed_file(root_dir, intermediate_dir, output):
    # get bfile for QCed SNPs in all eligible genes (output/G_SNP) and also G burden matrix w IID as index and genes as columns
    variant_gene = get_variant_exon_df(root_dir, output_csv=f'{intermediate_dir}/variant_gene.csv')  
    variant_gene['variant'].to_csv(f"{intermediate_dir}/ids.txt", index=False, header=False)
    args = [(c, root_dir, intermediate_dir) for c in range(1, 23)]
    with Pool(processes=8) as pool:       # adjust to available cores
        pool.starmap(run_plink, args)
    
    # 2) Make a merge-list of the remaining BED sets
    merge_list = f'{intermediate_dir}/merge_list.txt'
    with open(merge_list, 'w') as fh:
        for c in range(2, 23):  # use chr1 as the base
            prefix = f'{intermediate_dir}/chr{c}.preprune'
            fh.write(f'{prefix}.bed {prefix}.bim {prefix}.fam\n')

    # 3) Merge all chromosomes into one BED
    base_prefix = f'{intermediate_dir}/chr1.preprune'
    merged_prefix = f'{intermediate_dir}/allchr.preprune'
    cmd = f"""module load plink/1.9 && plink --bfile {base_prefix} \
        --merge-list {merge_list} --make-bed --out {merged_prefix}"""
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    # LD pruning & removing more genes with variant count less than 5
    result = subprocess.run(f'''module unload plink && module load plink/1.9 && plink --bfile {intermediate_dir}/allchr.preprune\
                            --indep-pairwise 50 5 0.5 --out {intermediate_dir}/allchr''', shell=True, capture_output=True, text=True, executable='/bin/bash')
    # final filter for variant count greater than 5 (read in prune in and subset where vc greater than 5)
    prune_in = pd.read_csv(f'{intermediate_dir}/allchr.prune.in',header=None,names=['variant'])
    variant_gene = variant_gene.merge(prune_in,how='inner',on='variant')
    vc = variant_gene['symbol'].value_counts()
    keep_genes = vc[vc > 5].index
    variant_gene = variant_gene[variant_gene['symbol'].isin(keep_genes)].copy()
    variant_gene['variant'].to_csv(f"{intermediate_dir}/ids_after_prune.txt", index=False, header=False)
    result = subprocess.run(f'module unload plink && module load plink/1.9 && plink --bfile {intermediate_dir}/allchr.preprune --extract {intermediate_dir}/ids_after_prune.txt --make-bed --recode A --out {output}_SNP', shell=True, capture_output=True, text=True, executable='/bin/bash')

    # get burden 
    G_SNP_raw = np.loadtxt(f'{output}_SNP.raw',  usecols=range(6, len(variant_gene['variant'].unique())+6), dtype=np.int64, skiprows=1)
    fam_df = pd.read_csv(f'{output}_SNP.fam',sep='\s+',header=None)
    fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
    iid_order = fam_df['IID'].values
    bim_df = pd.read_csv(f'{output}_SNP.bim',sep='\s+',header=None,names=['CHR','variant','CM','POS','A1','A2']).reset_index(drop=True)
    variant_order = bim_df['variant'].values
    G_SNP = pd.DataFrame(G_SNP_raw,columns=variant_order,index=iid_order)
    gene_to_vars = variant_gene.groupby("symbol")["variant"].apply(list)
    G = pd.DataFrame({gene: G_SNP[vars].sum(axis=1)
        for gene, vars in gene_to_vars.items()})
    G.to_csv(output)

def read_in_igsr_samples(igsr_samples_filepath,iid_order=None):
    # read in, subset to 2504, order correctly (if bfile path is not None)
    igsr_samples = pd.read_csv(igsr_samples_filepath,sep='\t')
    igsr_samples["Superpopulation code"] = igsr_samples["Superpopulation code"].str.split(",").str[0] # chose first for sample with EUR,AFR superpopulation code
    igsr_samples.rename(columns={'Sample name':'IID'},inplace=True)
    if iid_order is not None:
        igsr_samples = igsr_samples.set_index('IID').loc[iid_order].reset_index().rename(columns={'index':'IID'}).copy()
        assert igsr_samples['IID'].values.tolist()==iid_order
    return igsr_samples


def get_output_file_suffix(e,g_ps,c_ps,dataset):
    return f'e_{e}_g_ps_{g_ps}_c_ps_{c_ps}_dataset_{dataset}'

def rs(streams,stream_name):
        return int(streams[stream_name].integers(1, 2**31 - 1))

def seed_from(*xs):
    s = "|".join(f"{x:.8g}" if isinstance(x, float) else str(x) for x in xs)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)  # 32-bit int

def sun_generate_sim_data(G_path,igsr_samples_filepath,
                          output_dir,output_file_suffix,
                          g_ps,c_ps,g,e,M,num_clinical_assoc,num_genes, run_seed):
    '''
    Generate synthetic data similar to Sun et al. (Multi-view biclustering for genotype-phenotype association studies of complex diseases)
    using 1000 Genomes Phase 3 data. Use admixture files which contain 193634 markers with MAF>5% and 2504 individuals. 

    PARAMS:
    G_path: path to G file for genetic data
    igsr_samples_filepath: filepath corresponding to igsr samples data on superpopulation (downloaded from https://www.internationalgenome.org/data-portal/sample on 08/20/25)
    admixture_fractions_filepath: path
    output_file_suffix: such that if multiple simulations are created, each is distinctly defined - suffix for C and simulated data pkl file
    output_dir: where to write output genetic data matrix (X) in form of plink bfile, and clinical data matrix C
    M: number of clinical features (right now assuming all from one domain & all binary)
    g_ps: variable controlling how much population stratification is affecting genotype (value will determine quartile of AF variance (right now accepts 0.25 or 0.75)), or if 0 will pull from EUR superpopulation)
    c_ps: variable controlling the shift due to population stratification on phenotypic subgroup and clinical data matrix (value will determine range of uniform distribution shift Uniform(-a,a))
    (0-> pick SNPs in bottom 10% by allele frequency variance i.e. little pop. strat., 0.9-> pick SNPS in top 10% by allele frequency variance i.e. large pop. strat.)
    g: number of genes linked with subtype classification
    e: relative effect that genetic variation contributed to the effect of the phenotype. e in [0,1]. (decreased e means higher level of disagreement between genotypic and phenotypic subgroups)
    num_clinical_assoc: number of clinical features associated with subtype classification (same for all subtypes)
    num_genes: total number of genes (i.e. G.shape[1]])
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
    streams["genes"] =  np.random.default_rng(seed_from("genes")) # fix null markers regardless of g_ps
    streams["ps_noise"] = np.random.default_rng(seed_from("ps_noise")) # fix superpopulation shift in a subset defined by c_ps
    streams["clinical_weights"] =  np.random.default_rng(seed_from("clinical_weights")) 

    # read in G
    G = pd.read_csv(G_path,index_col=0)
    iid_order = G.index.values.tolist()

    # 1. Estimate gene burden variance across groups OR pull from superpopulation EUR only & generate true subgroups
    igsr_samples = read_in_igsr_samples(igsr_samples_filepath,iid_order)
    superpopulations = igsr_samples['Superpopulation code']

    genes_assoc_df = [] # names of the genes that are associated with each subgroup
    if g_ps !=0:
        assert g_ps in [0.25,0.75], f"right now code only takes top or bottom 25th percentile, value {g_ps} not accepted"
        # calculate weighted variance
        w = superpopulations.map(superpopulations.value_counts())
        w = w.astype(float).values
        w = w / w.sum()
        w = w.reshape(-1, 1)    
        # weighted mean for each gene
        X = G.values
        mu = (w * X).sum(axis=0) / w.sum()
        wvar = (w * (X - mu)**2).sum(axis=0) / w.sum()
        weighted_variance = pd.DataFrame(wvar, index=G.columns, columns=["weighted_variance"]).reset_index().rename(columns={'index': 'gene'})

        tenth_percentile = weighted_variance['weighted_variance'].quantile(0.1) 
        q1 = weighted_variance['weighted_variance'].quantile(0.25)
        q3 = weighted_variance['weighted_variance'].quantile(0.75)
        # Mark bottom/top quartiles; leave middle as NaN 
        weighted_variance["var_quartile"] = np.select([weighted_variance['weighted_variance'] <= q1, weighted_variance['weighted_variance'] >= q3],[0.25, 0.75],default=np.nan)
        weighted_variance["null_pool"] = weighted_variance['weighted_variance'] <= tenth_percentile
        # select genes (num_genes markers where half are in right af_var_quartile and half are from null pool)
        assert 3*g <= num_genes/2, f"{3*g} linked genes greater than {num_genes/2} genes to be pulled from null pool"
        selected_genes = []
        # generate genetic subgroups
        genetic_subgroup_dfs = []
        for genetic_subgroup in range(3):
            linked_genes = weighted_variance[weighted_variance["null_pool"]].sample(n=g, replace=False, random_state=rs(streams,"genes"))['gene'].values.tolist() 
            genes_assoc_df.append(pd.DataFrame({'genetic subgroup':genetic_subgroup, 'gene': linked_genes}))
            selected_genes.extend(linked_genes)
            s = G[linked_genes].sum(axis=1)
            genetic_subgroup_df = pd.DataFrame({'in_subgroup': (s >= s.quantile(0.8)).astype(int),'burden_sum': s, 'subgroup':genetic_subgroup})
            genetic_subgroup_dfs.append(genetic_subgroup_df)
        genetic_subgroup_dfs = pd.concat(genetic_subgroup_dfs)
        # draw such that half from null and half from stratified pool
        genes_assoc_df = pd.concat(genes_assoc_df)
        if num_genes//2 - len(genes_assoc_df['gene'].unique())>0:
            selected_genes.extend(weighted_variance[(weighted_variance["null_pool"])&(~weighted_variance['gene'].isin(selected_genes))]
                            .sample(n=num_genes//2-len(genes_assoc_df['gene'].unique()), replace=False, random_state=rs(streams,"genes"))['gene'].values.tolist()) # make sure half from null pool
        selected_genes.extend(weighted_variance[(weighted_variance['var_quartile']==g_ps)&(~weighted_variance['gene'].isin(selected_genes))] 
                        .sample(n=num_genes//2, replace=False, random_state=rs(streams,"genes"))['gene'].values.tolist()) # & half from correct pool 

    else:
        eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
        # keep only european samples
        G = G[G.index.isin(eur_samples)].copy()

        # select gene pool from which to draw linked genes (right now don't assert that MAF of these SNPs greater than 5% in EUR population also)
        gene_pool = G.columns.tolist()

        selected_genes = []
        # generate genetic subgroups
        genetic_subgroup_dfs = []
        for genetic_subgroup in range(3):
            linked_genes = streams["genes"].choice(gene_pool,size=g, replace=False).tolist()
            genes_assoc_df.append(pd.DataFrame({'genetic subgroup':genetic_subgroup, 'gene': linked_genes}))
            selected_genes.extend(linked_genes)
            s = G[linked_genes].sum(axis=1)
            genetic_subgroup_df = pd.DataFrame({'in_subgroup': (s >= s.quantile(0.8)).astype(int),'burden_sum': s, 'subgroup':genetic_subgroup})
            genetic_subgroup_dfs.append(genetic_subgroup_df)
        genetic_subgroup_dfs = pd.concat(genetic_subgroup_dfs)
        genes_assoc_df = pd.concat(genes_assoc_df)
        selected_genes.extend(streams["genes"].choice(list(set(gene_pool)-set(selected_genes)),size=num_genes-len(set(selected_genes)), replace=False).tolist()) 
    assert len(set(selected_genes)) == num_genes

    # keep individuals in exactly one subgroup
    keep_iids = (genetic_subgroup_dfs.loc[genetic_subgroup_dfs['in_subgroup'].eq(1)]
            .groupby(level=0).size().loc[lambda s: s==1].index)
    genetic_subgroup_dfs = genetic_subgroup_dfs.loc[keep_iids].copy()
    iid_order = [i for i in iid_order if i in genetic_subgroup_dfs.index]
    # write G
    G = G.loc[iid_order][selected_genes].copy()
    G.to_csv(f'{output_dir}/G_{output_file_suffix}')
    
    # 2. Generate phenotypic subgroups (concordance defined by e)
    genetic_subgroup_membership_matrix = genetic_subgroup_dfs.pivot(columns='subgroup',values='in_subgroup').to_numpy('float64')
    true_subgp = genetic_subgroup_membership_matrix.argmax(axis=1)  # vector with true genetic subgp (0,1,2)
    conf_matrix = np.full((3,3), (1-e)/(2))  
    np.fill_diagonal(conf_matrix, e) # confusion matrix with probability e phenotype subgroup agrees with genotype
    phenotype_labels = np.array([streams["env_noise"].choice(3, p=conf_matrix[g]) for g in true_subgp])
    phenotypic_membership_matrix = np.eye(3, dtype=int)[phenotype_labels]
    phenotypic_subgroup_dfs = pd.DataFrame(phenotypic_membership_matrix,index=iid_order,columns=[i for i in range(3)]).reset_index().rename(columns={'index':'IID'})
    phenotypic_subgroup_dfs = pd.melt(phenotypic_subgroup_dfs, id_vars='IID', var_name='subgroup', value_name='in_subgroup').set_index('IID')


    # 3. simulate M clinical features
    # start with baseline probabilities
    igsr_samples =  igsr_samples.set_index('IID').loc[iid_order].reset_index()
    Z = pd.get_dummies(igsr_samples['Superpopulation code']).to_numpy('float64')
    W = phenotypic_subgroup_dfs.pivot(columns='subgroup',values='in_subgroup')
    assert W.index.tolist()==iid_order
    W = W.to_numpy('float64')
    # generate superpop shift for each feature
    U_C = np.clip(streams["ps_noise"].normal(loc=1, scale=0.1, size=(M, Z.shape[1])), 0, None)
    # pick num_clinical_assoc linked features for each subgroup
    H_C = np.zeros((M,3), float)
    for j in range(3):
        idx = streams["assoc"].choice(M, size=num_clinical_assoc, replace=False)
        H_C[idx, j] = np.clip(streams["clinical_weights"].normal(loc=0.6, scale=0.1, size=len(idx)), 0, None)
    shared_factors = streams['env_noise'].normal(size=(len(iid_order), 3))  # 3 correlated latent sources
    A = streams['env_noise'].normal(scale=0.1, size=(3, M))                 # loading matrix
    C = streams["poisson"].poisson(np.exp(0.1 + W@H_C.T + (c_ps)*Z@U_C.T + shared_factors@A + streams["env_noise"].normal(0, 0.2, size=(len(iid_order), M))))
    clinical_assoc_df = pd.DataFrame(pd.DataFrame(H_C))

    # write C
    np.save(f"{output_dir}/C_{output_file_suffix}.npy", C)

    # write simulation metadata
    bundle = {
    "iid_order": iid_order,                          # list/array
    "genetic_subgroups": genetic_subgroup_dfs,          # pandas DataFrame
    "phenotypic_subgroups": phenotypic_subgroup_dfs,    # pandas DataFrame
    "genes_assoc": genes_assoc_df,                   # pandas DataFrame
    "clinical_assoc": clinical_assoc_df,             # pandas DataFrame
    "run_seed": run_seed                             # int
    }
    with open(f"{output_dir}/simulation_metadata_{output_file_suffix}.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return genetic_subgroup_dfs, phenotypic_subgroup_dfs, C, iid_order, genes_assoc_df, clinical_assoc_df 
