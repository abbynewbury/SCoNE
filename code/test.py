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
from itertools import product
from collections import defaultdict
import simulations.genomes1000_sim as sim_functions


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
bfile_path = f'{sim_output_dir}/G'
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
        assert all(self.bim_df['SNP'].values==G_columns)
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
        def test_non_overlapping_samples_in_subgroups(self):
            for file in files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                # make sure no overlapping iids in genetic subgroups
                genetic_subgroups = data["genetic_subgroups"]
                subgroup0 = set(genetic_subgroups[(genetic_subgroups['subgroup'])&(genetic_subgroups['genetic_subgroup']==0)]['IID'].values)
                subgroup1 = set(genetic_subgroups[(genetic_subgroups['subgroup'])&(genetic_subgroups['genetic_subgroup']==1)]['IID'].values)
                assert set(subgroup0).intersection(set(subgroup1)) == set()
                # make sure no overlapping iids in phenotypic subgroups
                phenotypic_subgroups = data["phenotypic_subgroups"]
                subgroup0 = set(phenotypic_subgroups[(phenotypic_subgroups['subgroup'])&(phenotypic_subgroups['phenotypic_subgroup']==0)]['IID'].values)
                subgroup1 = set(phenotypic_subgroups[(phenotypic_subgroups['subgroup'])&(phenotypic_subgroups['phenotypic_subgroup']==1)]['IID'].values)
                assert set(subgroup0).intersection(set(subgroup1)) == set()

    def test_markers_and_genetic_subgroup(self):
        files = sorted(glob.glob(os.path.join(sim_output_dir, "simulation_metadata_*.pkl")))
        for file in files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
            fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
            iid_index = fam_df[fam_df['IID'].isin(data['iid_order'])].index # some samples removed due to overlap btwn subgroups, need correct length
            
            g = int(re.search(r"markersassoc_(\d+)", file).group(1))
            
            genetic_subgroups = (data['genetic_subgroups'].pivot(index='IID',columns='genetic_subgroup',values='subgroup')
                .reindex(data['iid_order']).to_numpy().astype(int))
            genetic_subgroups_centered = (genetic_subgroups - genetic_subgroups.mean(axis=0)) / genetic_subgroups.std(axis=0, ddof=1)

            # correlation matrix: (n_genetic_subgroupscols × n_Gcols)
            G_centered = (self.G[iid_index,:] - self.G[iid_index,:].mean(axis=0)) / self.G[iid_index,:].std(axis=0, ddof=1)
            corr = genetic_subgroups_centered.T @ self.G_centered / (self.G.shape[0] - 1)
            for k,v in data["markers_assoc"].items():
                v_idx = self.bim_df[self.bim_df['SNP'].isin(v)].index.values.tolist() # get index of true linked markers
                v_corr = corr[k,v_idx].mean()
                rand_idx = np.random.choice([i for i in range(self.G.shape[1]) if i not in v_idx], size=g, replace=False) 
                rand_corr = corr[k,rand_idx].mean()
                assert v_corr>rand_corr
                
    def test_markers_and_phenotypic_subgroup(self):
        # genetic markers and phenotypic subgroup
        ps_list = [True,False]
        e_list = [0.25, 0.50, 0.75, 1]
        g_list = [100,500] # g represents the number of linked markers
        dataset_list = range(11) # 11 random datasets for each combination
        overall = []
        for ps, g, dataset in list(product(ps_list,g_list,dataset_list)):
            print(f'ps: {ps}, g:{g}, dataset:{dataset}')
            e_corr = defaultdict(list) # correlation of phenotypic gi w/ linked snps (take mean over two gi subgroups) for different e values
            for e in e_list:
                output_file_suffix = sim_functions.get_output_file_suffix(ps,e,dataset,g)
                with open(f"{sim_output_dir}/simulation_metadata_{output_file_suffix}.pkl", "rb") as f:
                    data = pickle.load(f)
                fam_df = pd.read_csv(f'{bfile_path}.fam',sep='\s+',header=None)
                fam_df.columns = ['FID','IID'] + fam_df.columns[2:].tolist()
                iid_index = fam_df[fam_df['IID'].isin(data['iid_order'])].index # some samples removed due to overlap btwn subgroups, need correct length
                    
                phenotypic_subgroups = (data['phenotypic_subgroups'].pivot(index='IID',columns='phenotypic_subgroup',values='subgroup')
                    .reindex(data['iid_order']).to_numpy().astype(int))
                phenotypic_subgroups_centered = (phenotypic_subgroups - phenotypic_subgroups.mean(axis=0)) / phenotypic_subgroups.std(axis=0, ddof=1)

                # correlation matrix: (n_phenotypic_subgroupscols × n_Gcols)
                G_centered = (self.G[iid_index,:] - self.G[iid_index,:].mean(axis=0)) / self.G[iid_index,:].std(axis=0, ddof=1)
                corr = phenotypic_subgroups.T @ G_centered / (self.G.shape[0] - 1)
                for k,v in data["markers_assoc"].items():
                    gi_idx = self.bim_df[self.bim_df['SNP'].isin(v)].index.values.tolist() # get index of true linked markers
                    e_corr[e].append(corr[k,gi_idx])
                    gi_corr = corr[k,gi_idx].mean()
                    rand_idx = np.random.choice([i for i in range(self.G.shape[1]) if i not in gi_idx], size=g, replace=False) 
                    rand_corr = corr[k,rand_idx].mean()
                    non_gi_corr = corr[k+2,gi_idx].mean()
                    # correlation w/ linked snps greater than random snps for genetically informed (gi) subgroup
                    assert gi_corr>rand_corr
                    # correlation of linked snps w/ gi subgroup greater than that with non-gi phenotypic subgroup (compare gi=0 with non-gi=2 and gi=1 with non-gi=3)
                    assert gi_corr>non_gi_corr
            overall.append(np.mean(np.concatenate(e_corr[1]))>np.mean(np.concatenate(e_corr[0.25])))
        # assert that more often than not, stronger correlation when e=1 vs. 0.25
        assert (sum(overall) / len(overall))>0.5
    
    

class TestSCoNE(unittest.TestCase):
    def setUp(self):
        self.rank = 3
        self.W = np.random.random((2500, self.rank))*1e-2+1e-6 
        self.H_G = np.random.random((200, self.rank))*1e-2+1e-6
        self.H_C = np.random.random((100, self.rank))*1e-2+1e-6
        self.U_G = np.random.random((200, 10))*1e-2+1e-6
        self.U_C = np.random.random((100, 10))*1e-2+1e-6
    
    def test_none_loss_type(self):
        G = np.random.choice([0, 1], (2500,200))
        C = np.random.poisson(lam=3, size=(2500,100))
        Z = np.random.poisson(lam=3, size=(2500,100))
        iid_index = range(2500)
        iid_index = range(2500)
        # if C_loss_type and G_loss_type are None, shouldn't be any change in factor matrices from random
        factor_matrices, loss_function = SCoNE.alternating_opt(G,C,np.zeros((2500,10)),
            self.W, self.H_G, self.H_C, self.U_G, self.U_C,lambda_W=0, lambda_H_G=0, lambda_H_C=0, 
            method='L-BFGS-B',options={'maxcor':10, 'maxiter':10, 'gtol':1e-5, 'maxls':20, 'ftol':1e-6},tol=1e-4,nonneg=True, C_loss_type=None, G_loss_type=None)
        all(np.isclose(factor_matrices["W"].flatten(), (self.W/np.linalg.norm(self.W, axis=0)).flatten()))

        # test when G loss is None
        factor_matrices, loss_dict = SCoNE.alternating_opt(G=G[iid_index,:], C=C, Z=Z, W=self.W, H_G=self.H_G, 
                                H_C=self.H_C, U_G=self.U_G, U_C=self.U_C,
                                lambda_W=0, lambda_H_G=0, lambda_H_C=0,
                                method='L-BFGS-B', options={'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6},  # keep scipy methods and options fixed
                                G_loss_type=None, C_loss_type='kl_div',
                                max_outer=50, min_outer=5, tol=1e-6, nonneg=True)
        ratios = factor_matrices["H_G"]/self.H_G
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert all(col_factors)
        # col_factors should not be true below since W, H_C was optimized
        ratios = factor_matrices["H_C"]/self.H_C
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)
        ratios = factor_matrices["W"]/self.W
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)
        ratios = factor_matrices["U_C"]/self.U_C
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)

        # test when C loss is None
        factor_matrices, loss_dict = SCoNE.alternating_opt(G=G[iid_index,:], C=C, Z=Z, W=self.W, H_G=self.H_G, 
                                H_C=self.H_C, U_G=self.U_G, U_C=self.U_C,
                                lambda_W=0, lambda_H_G=0, lambda_H_C=0,
                                method='L-BFGS-B', options={'maxcor':10,'maxiter':10,'gtol':1e-5,'maxls':5,'ftol':1e-6},  # keep scipy methods and options fixed
                                G_loss_type='bce', C_loss_type=None,
                                max_outer=50, min_outer=5, tol=1e-6, nonneg=True)
        ratios = factor_matrices["H_C"]/self.H_C
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert all(col_factors)
        # col_factors should not be true below since W, H_G was optimized
        ratios = factor_matrices["H_G"]/self.H_G
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)
        ratios = factor_matrices["W"]/self.W
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)
        ratios = factor_matrices["U_G"]/self.U_G
        col_factors = [np.allclose(ratios[0,j],ratios[:, j]) for j in range(ratios.shape[1])]
        assert not all(col_factors)


# test mlflow setup:
# g = df.groupby("experiment_id")[["params.ps", "params.e", "params.num_markers_assoc"]].nunique()
# assert g[g.ne(1).any(axis=1)].shape[0] == 0 # assert that for every unique experiment id, only one ps, e, num_markers_assoc value

# test cluster eval metrics
# current test embedded in nmi, that our nmi calculation matches sklearn nmi

# test that all fam files for the same G have iid in same order (Verify map and ped root dir files have same order as fam once make bfile)

# test that the sigma for MVBC really does minimize the objective function

# make sure SCoNE chooses init with lowest loss