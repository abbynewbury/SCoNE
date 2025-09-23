import unittest
import numpy as np
import algorithms.SCoNE as SCoNE
import pickle
import numpy as np
import pandas as pd
import os
import glob
import hashlib
import re

# DEFINE PATHS
intermediate_plink_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_plink'
intermediate_saige_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/intermediate_saige'
root_dir = '/gpfs/commons/datasets/1000genomes'
sim_output_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/output'
graph_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/output'
igsr_samples_filepath = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/data/simulations/input/igsr_samples.tsv'
maf_by_superpop_filepath = f'{intermediate_plink_dir}/maf_by_superpop'
admixture_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5'
map_filepath = f'{root_dir}/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.map'
code_dir = '/gpfs/commons/groups/gursoy_lab/anewbury/unsupervised_pheno/code'
# DEFINE PATHS

# no matrices C are identical
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class TestSimulatedData(unittest.TestCase):
    def setUp(self):
        # read in 
        G = np.loadtxt(f'{sim_output_dir}/G.raw',  usecols=range(6, 10000+6), dtype=np.int8, skiprows=1)
        self.G = (G>0).astype(np.int8)
        # assert that bim_df and G order is the same
        bfile_path = f'{sim_output_dir}/G'
        self.bim_df = pd.read_csv(f'{bfile_path}.bim',sep='\s+',header=None,names=['CHR','SNP','CM','POS','A1','A2']).reset_index(drop=True)
        G_columns = [i.split('_')[0] for i in pd.read_csv(f'{bfile_path}.raw',sep='\s+',usecols=range(6, 10000+6),nrows=1).columns]
        assert all(bim_df['SNP'].values==G_columns)
    def tearDown(self):
        pass
    def test_randomness(self):
        # test no two datasets are the same 
        # no two clinical matrices are same
        files = sorted(glob.glob(os.path.join(sim_output_dir, "C_*.npy")))
        assert len(files) == 1616
        hashes = {file_hash(f): f for f in files}
        if len(hashes) != len(files):
            raise AssertionError("Duplicate content detected")
        
        files = sorted(glob.glob(os.path.join(sim_output_dir, "simulation_metadata_*.pkl")))
        # linked markers and clinical indices aren't the same
        markers_seen = {} # dict for linked markers
        clinical_indices_seen = {} # dict for linked clinical variables
        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            for k,v in data["markers_assoc"].items():
                markers = tuple(sorted(v))
                if k not in markers_seen: markers_seen[k] = set()
                if markers in markers_seen[k]:
                    raise AssertionError(f"Duplicate subgroup {k} markers in {file}")
                markers_seen[k].add(markers)
            for k in data['clinical_assoc']['phenotypic_subgroup'].unique():
                clinical_indices = tuple(sorted(np.concatenate(data['clinical_assoc'].set_index('phenotypic_subgroup').loc[0]['indices'].values)))
                if k not in clinical_indices_seen: clinical_indices_seen[k] = set()
                if clinical_indices in clinical_indices_seen[k]:
                    raise AssertionError(f"Duplicate subgroup {k} clinical indices in {file}")
                clinical_indices_seen[k].add(clinical_indices)

    def test_markers_and_genetic_subgroup(self):
        files = sorted(glob.glob(os.path.join(sim_output_dir, "simulation_metadata_*.pkl")))
        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            g = int(re.search(r"markersassoc_(\d+)", file).group(1))
            
            genetic_subgroups = (data['genetic_subgroups'].pivot(index='IID',columns='genetic_subgroup',values='subgroup')
                .reindex(data['iid_order']).to_numpy().astype(int))
            G_centered = (self.G - self.G.mean(axis=0)) / self.G.std(axis=0, ddof=1)
            genetic_subgroups_centered = (genetic_subgroups - genetic_subgroups.mean(axis=0)) / genetic_subgroups.std(axis=0, ddof=1)

            # correlation matrix: (n_genetic_subgroupscols × n_Gcols)
            corr = genetic_subgroups_centered.T @ G_centered / (G.shape[0] - 1)
            for k,v in data["markers_assoc"].items():
                v_idx = self.bim_df[self.bim_df['SNP'].isin(v)].index.values.tolist() # get index of true linked markers
                v_corr = corr[k,v_idx].mean()
                rand_idx = np.random.choice([i for i in range(G.shape[1]) if i not in v_idx], size=g, replace=False) 
                rand_corr = corr[k,rand_idx].mean()
                assert v_corr>rand_corr
    
    

class TestSCoNE(unittest.TestCase):
    def setUp(self):
        self.rank = 3
        self.W = np.random.random((2500, 3))*1e-2+1e-6 
        self.H_G = np.random.random((200, 3))*1e-2+1e-6
        self.H_C = np.random.random((100, 3))*1e-2+1e-6
        self.U_G = np.random.random((200, 10))*1e-2+1e-6
        self.U_C = np.random.random((100, 10))*1e-2+1e-6
    
    def test_none_loss_type(self):
        G = np.random.choice([0, 1], (2500,200))
        C = np.random.poisson(lam=3, size=(2500,100))
        # if C_loss_type and G_loss_type are None, shouldn't be any change in factor matrices from random
        factor_matrices, loss_function = SCoNE.alternating_opt(G,C,np.zeros((2500,10)),
            self.W, self.H_G, self.H_C, self.U_G, self.U_C,lambda_W=0, lambda_H_G=0, lambda_H_C=0, 
            method='L-BFGS-B',options={'maxcor':10, 'maxiter':10, 'gtol':1e-5, 'maxls':20, 'ftol':1e-6},tol=1e-4,nonneg=True, C_loss_type=None, G_loss_type=None)
        all(np.isclose(factor_matrices["W"].flatten(), (self.W/np.linalg.norm(self.W, axis=0)).flatten()))

# test mlflow setup:
# g = df.groupby("experiment_id")[["params.ps", "params.e", "params.num_markers_assoc"]].nunique()
# assert g[g.ne(1).any(axis=1)].shape[0] == 0 # assert that for every unique experiment id, only one ps, e, num_markers_assoc value