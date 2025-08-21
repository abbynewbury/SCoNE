import pandas as pd
import subprocess
from functools import reduce
import numpy as np
import umap
from plotnine import *
import os
from simulations.genomes1000_sim import *

# DEFINE PATHS
intermediate_file_dir ='/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
root_dir = '/gpfs/commons/datasets/1000genomes'
output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
igsr_samples_path = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
# DEFINE PATHS


igsr_samples = pd.read_csv(igsr_samples_path,sep='\t')
igsr_samples = igsr_samples[igsr_samples['Sample name'].isin(iid_order)].copy()
igsr_samples["Sample name"] = pd.Categorical(igsr_samples["Sample name"], categories=iid_order, ordered=True)
igsr_samples = igsr_samples.sort_values("Sample name").reset_index(drop=True)
igsr_samples.head()