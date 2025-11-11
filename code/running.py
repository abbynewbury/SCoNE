# running stuff to merge chrs in vcf file and filter etc.

from cyvcf2 import VCF
import pandas as pd
import subprocess
import numpy as np
import simulations.genomes1000_sim as sim_functions
from multiprocessing import Pool


# TO RUN BEFOREHAND 
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
    exon_terms = ['missense_variant','synonymous_variant','stop_gained','stop_lost','start_lost','frameshift_variant','inframe_insertion','inframe_deletion','coding_sequence_variant']
    variant_gene = variant_gene[variant_gene['consequence'].isin(exon_terms)].copy()
    # avoid duplicate variant-gene pairs
    variant_gene = variant_gene.drop_duplicates(['variant','allele']).copy()
    # only keep those genes with variant count above 5    
    vc = variant_gene['symbol'].value_counts()
    keep_genes = vc[vc > 5].index
    variant_gene = variant_gene[variant_gene['symbol'].isin(keep_genes)].copy()

    variant_gene.to_csv(output_csv,sep='\t')
    return variant_gene


def prep_1000genomes_bed_file(root_dir, intermediate_dir, output):
    # get bfile for QCed SNPs in all eligible genes (output/G_SNP) and also G burden matrix w IID as index and genes as columns
    variant_gene = get_variant_exon_df(root_dir, output_csv=f'{intermediate_dir}/variant_gene.csv')  
    variant_gene['variant'].to_csv(f"{intermediate_dir}/ids.txt", index=False, header=False)
    def run_plink(chr):
        cmd = f'''module load plink/1.9 && plink --vcf {root_dir}/GRCh37/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz\
                --extract {intermediate_dir}/ids.txt\
                --biallelic-only strict --maf 0.05 --geno 0.05 --mind 0.05 --make-bed --out {intermediate_dir}/chr{chr}.preprune'''
        result = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash") 
    with Pool(processes=8):       # adjust to available cores
        Pool().map(run_plink, range(1, 23))
    
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


# TO RUN BEFOREHAND

# # if g_ps = 0
# igsr_samples = sim_functions.read_in_igsr_samples(igsr_samples_filepath)
# eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
# # keep for european samples
# with open(f'{intermediate_plink_dir}/eursamples.txt','w') as f:
#     for iid in eur_samples:
#         f.write(iid + "\t" + iid + "\n")
# keep_samples = f'--keep {intermediate_plink_dir}/eursamples.txt'

