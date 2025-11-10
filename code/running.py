# running stuff to merge chrs in vcf file and filter etc.

from cyvcf2 import VCF
import pandas as pd
import subprocess
import numpy as np
import simulations.genomes1000_sim as sim_functions
intermediate_plink_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
root_dir = '/gpfs/commons/datasets/1000genomes'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'



rows = []
for chr in range(1,23):
    vcf_path = f"{root_dir}/release-20130502-supporting/functional_annotation/unfiltered/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5_func_anno.20130502.sites.vcf.gz"
    vcf = VCF(vcf_path)
    hdr = vcf.raw_header
    csq_format = vcf.get_header_type('CSQ')['Description']
    fields = csq_format.split("Format: ")[1].strip('"').split('|')
    EXONIC_TERMS = {
        "missense_variant", "synonymous_variant", "stop_gained",
        "stop_lost", "start_lost", "frameshift_variant",
        "inframe_insertion", "inframe_deletion",
        "coding_sequence_variant"
    }
    idx = {f:i for i,f in enumerate(fields)}
    count = 0
    for rec in vcf:
        count+=1
        csq = rec.INFO.get("CSQ")
        for entry in csq.split(","):
                parts = entry.split("|")
                consequence = parts[idx["Consequence"]]
                if any(term in consequence.split("&") for term in EXONIC_TERMS):
                    gene = parts[idx["SYMBOL"]] or parts[idx["Gene"]]
                    allele = parts[idx["Allele"]]
                    if rec.ID and rec.ID != ".":
                        var_id = rec.ID
                    else:
                        var_id = f"{rec.CHROM}:{rec.POS}:{rec.REF}:{allele}"
                    rows.append({"variant": var_id, "gene": gene, "allele":allele})

df = pd.DataFrame(rows).drop_duplicates()
# avoid duplicate variant-gene pairs
df = df.drop_duplicates(['variant','allele']).copy()
# write to csv
rsids = df["variant"].astype(str)
rsids = rsids[rsids.str.startswith("rs")]
rsids.to_csv("/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink/ids.txt", index=False, header=False)
variant_to_gene = df.set_index('variant')['gene'].to_dict()

igsr_samples = sim_functions.read_in_igsr_samples(igsr_samples_filepath)
eur_samples = igsr_samples[igsr_samples['Superpopulation code']=='EUR']['IID'].values.tolist()
# keep for european samples
with open(f'{intermediate_plink_dir}/eursamples.txt','w') as f:
    for iid in eur_samples:
        f.write(iid + "\t" + iid + "\n")
keep_samples = f'--keep {intermediate_plink_dir}/eursamples.txt'

for chr in range(1, 23):
    cmd = f'''module load plink/1.9 && plink --vcf {root_dir}/GRCh37/ALL.chr{chr}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz\
            --extract {intermediate_plink_dir}/ids.txt\
            {keep_samples}\
            --biallelic-only strict --maf 0.05 --geno 0.05 --mind 0.05 --make-bed --out {intermediate_plink_dir}/chr{chr}.preprune'''
    result = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash") 

# 2) Make a merge-list of the remaining BED sets (triplets required by PLINK are safest)
merge_list = f'{intermediate_plink_dir}/merge_list.txt'
with open(merge_list, 'w') as fh:
    for c in range(2, 23):  # use chr1 as the base
        prefix = f'{intermediate_plink_dir}/chr{c}.preprune'
        fh.write(f'{prefix}.bed {prefix}.bim {prefix}.fam\n')

# 3) Merge all chromosomes into one BED
base_prefix = f'{intermediate_plink_dir}/chr1.preprune'
merged_prefix = f'{intermediate_plink_dir}/allchr.preprune'
cmd = f"""module load plink/1.9 && plink --bfile {base_prefix} \
    --merge-list {merge_list} --make-bed --out {merged_prefix}"""
subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

# LD pruning
result = subprocess.run(f'module unload plink && module load plink/1.9 && plink --bfile {intermediate_plink_dir}/allchr.preprune --indep-pairwise 50 5 0.5 --out {intermediate_plink_dir}/allchr', shell=True, capture_output=True, text=True, executable='/bin/bash')
result = subprocess.run(f'module unload plink && module load plink/1.9 && plink --bfile {intermediate_plink_dir}/allchr.preprune --extract {intermediate_plink_dir}/allchr.prune.in --make-bed --out {intermediate_plink_dir}/allchr', shell=True, capture_output=True, text=True, executable='/bin/bash')