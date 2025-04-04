# Generate simulated data as in Dahl et al. Reverse GWAS paper
import numpy as np
import pandas as pd
from scipy import stats


def generate_sim_data(N,S,Q,K,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,num_pops):
    #TODO: from paper, 'by default SNPs have same type of effect on all traits' - can change
    '''
    N: sample size
    S: # SNPs
    Q: # traits
    K: # subtypes
    p: proportion in each subtype (len(p) == K)
    s: number of SNPs with no effect, homogeneous effect, heterogeneous effect [S_null, S_hom, S_het]
    pge: amount of correlation between subtype state z and snp genotypes in G (0 - z independent of genotype -- 1 - completely heritable)
    snp_hom_effects: large or moderate SNP effects
    snps_af_range: list of range for SNP allele frequencies
    mus_variance: variance on subtype main effects on trait
    num_pops: number of subpopulations (population stratification)
    '''
    assert len(p) == K # need the proportions for each subtype to be the same as # subtypes
    assert sum(p) == 1 # and add to 1
    assert len(p) == 2, "Did you change the  model to allow for hetergeneous effects in other groups now? - need to change G-E correlation then too and remove p, etc."
    S_null = s[0]
    S_hom = s[1]
    S_het = s[2]
    pop_main_effects_var = .05 # fixed to .1 for now

    #snps_af = np.random.uniform(snps_af_range[0],snps_af_range[1],S) # drawing allele freq. pi from Uniform[0.05,0.5] - bias the SNPs towards being larger 
    # equal proportions in each sub-population
    sub_pops = [i for i in range(1, num_pops) for _ in range(N // num_pops)] + [num_pops]*((N // num_pops)+ (N % num_pops))
    snps = []
    for pop in range(1,num_pops+1):
        pi = stats.beta.rvs(4.5, 4.5, size=S)
        snps_pop = np.random.binomial(2, pi, (len([i for i in sub_pops if i==pop]), S)) 
        snps.append(snps_pop)
    snps = np.vstack(snps)

    snp_types = np.array(['null']*S_null + ['hom']*S_hom + ['het']*S_het)
    mask_hom = snp_types == 'hom'
    mask_het = snp_types == 'het'

    # generate SNP effects - betas in KxSXQ 
    if snp_hom_effects == 'large':
        sigma2_hom = 0.04
    else:
        sigma2_hom = 0.004
    sigma2_het = 0.044 - sigma2_hom
    assert sigma2_het>0
    
    # Homogeneous SNPs
    alpha = np.random.normal(
        0, np.sqrt(sigma2_hom / S_hom), (S_hom, Q)) 

    # Heterogeneous SNPs
    betas = {}
    beta = np.random.normal(
    0, np.sqrt(sigma2_het / (p[0]*S_het)), (S_het,  Q)) # only acting in gp 1 for now
    betas[0] = beta
    betas[1] = np.zeros((S_het,  Q))

    # Null SNPs remain 0

    # subtype state z is correlated with SNP genotypes in G (dependent on pge parameter)
    omega = np.random.normal(0, 1/S_het, (S_het,))  # omega ~ N(0, 1/S)
    delta = np.random.normal(0, 1, (N,)) 
    z_tilde = np.sqrt(pge) * (snps[:,mask_het] @ omega) + np.sqrt(1 - pge) * delta
    tau = np.percentile(z_tilde, 100 * (1 - p[0])) # ensure p split
    z = (z_tilde > tau).astype(int)
    assert np.isclose(np.mean(z),p[0])

    # Add main subtype effects (mu)
    mus = np.random.normal(0,mus_variance,(K, Q))

    pop_main_effects = np.random.normal(0,pop_main_effects_var,(K, Q))
    pop_main_effects = []
    for pop in range(1,num_pops+1):
        pop_main_effects_mean = stats.beta.rvs(10, 4)
        pop_main_effects_pop = np.random.normal(pop_main_effects_mean,pop_main_effects_var,(len([i for i in sub_pops if i==pop]), Q))
        pop_main_effects.append(pop_main_effects_pop)
    pop_main_effects = np.vstack(pop_main_effects)

    # record metadata
    snp_metadata_lst = [pd.DataFrame(np.zeros((S_null, Q))),pd.DataFrame(alpha), pd.DataFrame(beta)]
    snp_metadata = pd.concat(snp_metadata_lst).reset_index(drop=True)
    snp_metadata['type'] = ['null']*S_null + ['hom']*S_hom +['het']*S_het
    trait_metadata = pd.DataFrame(mus.T, columns=['subtype1','subtype2'])

    # simulate quantitative phenotypes using these covariates
    Y0 = np.empty((N, Q))
    mus_ = []
    pop_main_effects_ = []
    het = []
    hom = []
    noise = []
    for i in range(N):
        Y0[i, :] = mus[z[i], :] + pop_main_effects[i, :] + snps[i,mask_hom] @ alpha + snps[i,mask_het] @ betas[z[i]] + (0.25)*np.random.randn(Q) # subtype main effects + pop. main effects + homogeneous effects + heterogeneous effects
        mus_.extend(mus[z[i], :].flatten().tolist())
        pop_main_effects_.extend(pop_main_effects[i, :].flatten().tolist())
        het.extend((snps[i,mask_het] @ betas[z[i]]).flatten().tolist())
        hom.extend((snps[i,mask_hom] @ alpha).flatten().tolist())
        noise.extend(((0.25)*np.random.randn(Q)).flatten().tolist())  
    # if  below a certain threshold, re-adjust to label as control - maybe don't need to
    true_subtypes = pd.DataFrame(z,columns=['true subtype'])
    true_subpops = pd.DataFrame(sub_pops,columns=['true subpop'])
    return Y0, snps, snp_metadata, trait_metadata, true_subtypes,true_subpops

# Use: C,X,snp_metadata, trait_metadata, true_subtypes,true_subpops = generate_sim_data(N,S,Q,K,p,s,pge,snp_hom_effects,snps_af_range,mus_variance,num_pops)