import numpy as np
try:
    import cupy as cp # if GPU
except ImportError:
    cp = None
from collections import defaultdict
from scipy.special import xlogy
from joblib import Parallel, delayed
import json
import pickle
from ._initialize_nmf import _initialize_nmf
from ..evaluation.reconstruction_evaluation import calculate_ccc

def compute_loss(X,X_hat,loss_type):
    if loss_type == 'kl_div':
        return np.sum(-xlogy(X,X_hat) + xlogy(X,X) + X_hat - X) 
        #return np.sum(-np.multiply(X,np.log(X_hat)) + X_hat + gammaln(X + 1)) # log(X!) can help stabilize/make pos.
    elif loss_type == 'fro':
        return (1/2)*np.dot((X - X_hat).ravel(), (X - X_hat).ravel())
    elif loss_type is None:
        return 0
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'fro', None"

def l1_norm(x):
    if x is not None:
        return np.sum(np.abs(x),dtype=np.float64)
    else:
        return 0

def l2_norm_squared(x):
    return np.sum(x * x, dtype=np.float64)

def compute_jac(sample_matrix, X, X_hat, loss_type, for_W=False):
    '''
    This function works to compute the Jacobian (unflattened) given loss type in  'kl_div', 'fro'
    Sample matrix defines the matrix that has N rows, will be W or Z -- helps this function to generalize to jac for H and U
    for_W should only be used when computing the Jacobian for the W matrix (puts sample matrix on other side/diff dimensions)
    '''
    if loss_type == 'kl_div':
        E_x = np.ones(X.shape)
        X_tilde = np.divide(X,X_hat)
        if for_W:
            GX = (E_x - X_tilde)@sample_matrix
        else:
            GX = (sample_matrix.T@(E_x - X_tilde)).T 
    elif loss_type == 'fro':
        if for_W:
            GX = (X_hat-X)@sample_matrix
        else:
            GX = (sample_matrix.T@(X_hat-X)).T
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'fro'"
    return GX

def get_X_hat(W,H,Z,U,loss_type):
    if Z is not None:
        X_hat = W@H.T + Z@U.T
    else:
        X_hat = W@H.T
    if loss_type in ['kl_div','fro']:
        X_hat = np.clip(X_hat, 1e-9, np.inf) # clipping for log purposes
    elif loss_type is None:
        X_hat = None
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'fro'"
    return X_hat

def total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, alpha, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # define nec. elements
    if G_loss_type is not None:
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        G_loss = compute_loss(G,G_hat,loss_type=G_loss_type)
    else: G_loss = 0
    if C_loss_type is not None:
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        C_loss = compute_loss(C,C_hat,loss_type=C_loss_type)
    else: C_loss = 0

    l2_regularization = alpha*l2_norm_squared(W)
    l1_regularization = lambda_H_G*l1_norm(H_G) +  lambda_H_C*l1_norm(H_C) # l1 reg.

    if G_loss < 0: assert True == False, f"G_loss with loss type {G_loss_type} negative"
    if C_loss < 0: assert True == False, f"C_loss with loss type {C_loss_type} negative"
    loss = lambda_Gloss*G_loss + C_loss + l2_regularization + l1_regularization
    return loss, lambda_Gloss*G_loss, C_loss, l2_regularization, l1_regularization


def make_fg_W(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, alpha, G_loss_type, C_loss_type):
    # Precompute terms independent of W
    def _forwardG(x):
        W = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        return G_hat
    def _forwardC(x):
        W = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
        W = x.reshape(shape, order='F')
        
        if G_loss_type is not None:
            G_hat = _forwardG(x)
            G_loss = compute_loss(G,G_hat,G_loss_type)
        else:
            G_loss = 0

        if C_loss_type is not None:
            C_hat = _forwardC(x)
            C_loss =  compute_loss(C,C_hat,C_loss_type)
        else:
            C_loss = 0
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss + (alpha / 2)*l2_norm_squared(W) # last term adds L2 norm on W
        return loss

    def g(x):
        W = x.reshape(shape, order='F')

        if G_loss_type is not None:
            G_hat = _forwardG(x)
            # compute jac
            GW_wrt_G = compute_jac(H_G, G, G_hat, G_loss_type, for_W=True)
        else:
            GW_wrt_G = np.zeros((W.shape[0],W.shape[1]))

        if C_loss_type is not None:
            C_hat = _forwardC(x)
            # compute jac
            GW_wrt_C = compute_jac(H_C, C, C_hat, C_loss_type, for_W=True)
        else:
            GW_wrt_C = np.zeros((W.shape[0],W.shape[1]))


        GW = (lambda_Gloss*GW_wrt_G + GW_wrt_C + alpha*W) # last term is gradient of L2 norm
        return GW.flatten(order='F')

    return f,g

def make_fg_HG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of H_G
    if C_loss_type is not None:
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        C_loss = compute_loss(C,C_hat,C_loss_type)
    else: C_loss = 0
    def _forward(x):
        H_G = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type) # could cut down on matrix multiplications if use precalculated Z@U_G.T for fun and jac
        return G_hat
    def f(x):
        G_hat = _forward(x)
        G_loss = compute_loss(G,G_hat,G_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss
        return loss
    def g(x):
        G_hat = _forward(x)
        # compute jac
        GH_G = lambda_Gloss*compute_jac(W, G, G_hat, G_loss_type)
        return GH_G.flatten(order='F')

    return f,g

def make_fg_HC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of H_C
    if G_loss_type is not None:
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        G_loss = compute_loss(G,G_hat,G_loss_type)
    else: G_loss = 0
    def _forward(x):
        H_C = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
        C_hat = _forward(x)
        C_loss = compute_loss(C,C_hat,C_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss
        return loss
    def g(x):
        C_hat = _forward(x)
        # compute jac
        GH_C = compute_jac(W, C, C_hat, C_loss_type)
        return GH_C.flatten(order='F')

    return f,g

def make_fg_UG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of U_G
    if C_loss_type is not None:
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        C_loss = compute_loss(C,C_hat,C_loss_type)
    else: C_loss = 0
    def _forward(x):
        U_G = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        return G_hat
    def f(x):
        G_hat = _forward(x)
        G_loss =  compute_loss(G,G_hat,G_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss
        return loss
    def g(x):
        G_hat = _forward(x)
        # compute jac
        GU_G = lambda_Gloss*compute_jac(Z, G, G_hat, G_loss_type)
        return GU_G.flatten(order='F')

    return f,g

def make_fg_UC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of U_C
    if G_loss_type is not None:
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        G_loss = compute_loss(G,G_hat,G_loss_type)
    else: G_loss = 0
    def _forward(x):
        U_C = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
        C_hat = _forward(x)
        C_loss = compute_loss(C,C_hat,C_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss
        return loss
    def g(x):
        C_hat = _forward(x)
        # compute jac
        GU_C = compute_jac(Z, C, C_hat, C_loss_type)
        return GU_C.flatten(order='F')

    return f,g

def soft_thresh(z, tau):
    # elementwise prox for L1: sign(z)*max(|z|-tau,0)
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

def proj_nonneg(x):
    # If already feasible, return x as-is (avoids an allocation most iterations)
    if x.min() >= 0:
        return x
    return np.maximum(x, 0)

def armijo_suff_decrease_cond(f_new,f,g,x_new,x,sigma):
    return f_new - f <= sigma*np.dot(g.ravel(),(x_new-x).ravel())

def pgd_armijo(fun, grad, x0, max_iter=500, rho=0.1, sigma=1e-4, ftol=1e-12, l1=0,max_ls=10):
    """
    Projected gradient descent with Armijo rule.

    f: objective function, g: gradient
    sigma (Armijo constant in (0,1)); rho (shrink factor in (0,1))
    l1: lambda for L1 soft-thresholding sparsity parameter
    max_ls: how much willing to increase/decrease step size by (rho^max_ls)
    """
    x = proj_nonneg(x0)
    n=1

    def x_and_s(step, x, g):
        z = x - step * g
        if l1 > 0:
            z = soft_thresh(z, step * l1)
        x_new = proj_nonneg(z)
        s = x_new - x
        return x_new, s

    # Armijo backtracking
    for it in range(max_iter):
        f= fun(x)
        g = grad(x)

        x_new, s = x_and_s(n, x, g)
        f_new = fun(x_new)

        if f_new - f <= sigma * np.vdot(g, s):
            # grow n: n <- n / rho until Armijo fails or projection makes no change
            for _ in range(max_ls):
                n_next = min(n / rho, 1e6) # cap growth
                x_next, s_next = x_and_s(n_next, x, g)
                if np.allclose(x_next, x_new):
                    break
                f_next = fun(x_next)
                if not (f_next - f <= sigma * np.vdot(g, s_next)):
                    break
                n, x_new, s, f_new = n_next, x_next, s_next, f_next
        else:
            for _ in range(max_ls):
                n_next= n * rho
                x_next, s_next = x_and_s(n_next, x, g)
                f_next = fun(x_next)
                if f_next - f <= sigma * np.vdot(g, s_next):
                    n, x_new, s, f_new = n_next, x_next, s_next, f_next
                    break
                n = n_next
            else:
                n=1 # reset, no acceptable step found within max_ls
                break
        
        # implement early stopping
        if abs(f_new - f)/max(1.0, abs(f)) < ftol:
            return x_new
        
        x = x_new
    return x


def alternating_opt(
    G,C,Z,              # true matrices
    rank,
    alpha=0,lambda_H_G=0, lambda_H_C=0, lambda_Gloss=1,                  # regularization parameters
    G_loss_type='kl_div', C_loss_type='kl_div', # in 'kl_div',  'fro' or None (None indicates not fitting to data, i.e. if G_loss_type=None & C_loss_type='fro then C only optimization)
    init='random', # using sklearn _initialize_nmf  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    max_inner=50, # options for pgd_armijo
    rho=0.1, # options for pgd_armijo (n shrink factor)
    sigma=1e-4, # options for pgd_armijo (Armijo constant)
    inner_ftol=1e-5, # options for pgd_armijo (stopping criteria for ftol)
    max_outer=200,
    min_outer=5,
    tol=1e-6,max_ls=50,post_hoc_rescale=False,
    # if test is True, W is the only factor matrix that will be optimized (for train/test split)
    test=False,
    H_G=None, H_C=None, U_G=None, U_C=None
):
    def one_block_update(name, X):
        x0 = X.flatten(order='F') 

        if name == "W":
            f,g = make_fg_W(W.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, alpha, G_loss_type, C_loss_type)
            l1 = 0
        elif name == "H_G":
            f,g = make_fg_HG(H_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = lambda_H_G
        elif name == "H_C":
            f,g = make_fg_HC(H_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = lambda_H_C
        elif name == "U_G":
            f,g = make_fg_UG(U_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = 0
        elif name == "U_C":
            f,g = make_fg_UC(U_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = 0
        else:
            raise ValueError(f"Unknown block {name}")
        x = pgd_armijo(f, g, x0, max_iter=max_inner, rho=rho, sigma=sigma, ftol=inner_ftol, l1=l1, max_ls=max_ls)
        return x.reshape(X.shape, order='F')

    N = G.shape[0] if G is not None else C.shape[0]

    # initialize factor matrices
    if test:
        W = np.random.uniform(low=0.1,high=1,size=(N, rank))
    else:
        W_init = []
        if G is not None:
            W_G, H_G = _initialize_nmf(G, n_components=rank, init=init)
            if np.mean(W_G == 0)>0 or np.mean(H_G == 0)>0:
                assert True==False, f"getting sparse factor matrix inits W_G:{np.mean(W_G == 0)}, H_G: {np.mean(H_G == 0)}. make sure C and G have type float"
            H_G = H_G.T
            W_init.append(W_G)
            if Z is not None:
                U_G = H_G.mean()*np.abs(np.random.standard_normal(size=(G.shape[1], Z.shape[1])))
            else: U_G = None
        else:
            H_G = None
            U_G = None
        if C is not None:
            W_C, H_C = _initialize_nmf(C, n_components=rank, init=init)
            if np.mean(W_C == 0)>0 or np.mean(H_C == 0)>0:
                assert True==False, f"getting sparse factor matrix inits W_C:{np.mean(W_C == 0)}, H_C: {np.mean(H_C == 0)}. make sure C and G have type float"
            H_C = H_C.T
            W_init.append(W_C)
            if Z is not None:
                U_C = H_C.mean()*np.abs(np.random.standard_normal(size=(C.shape[1], Z.shape[1])))
            else: U_C = None
        else:
            H_C = None
            U_C = None
        W = np.mean(W_init, axis=0)


    # initial objective
    loss_dict = defaultdict(list)
    f_prev, G_loss, C_loss, l2_regularization, l1_regularization = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, alpha, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
    for _ in range(max_outer):
        W   = one_block_update("W", W)
        if G_loss_type is not None and not test:
            H_G = one_block_update("H_G", H_G)
            if Z is not None:
                U_G = one_block_update("U_G", U_G)
        if C_loss_type is not None and not test:
            H_C = one_block_update("H_C", H_C)
            if Z is not None:
                U_C = one_block_update("U_C", U_C)

        f_cur, G_loss, C_loss, l2_regularization, l1_regularization = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, alpha, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
        loss_dict['total_loss'].append(f_cur)
        loss_dict['G_loss'].append(G_loss)
        loss_dict['C_loss'].append(C_loss)
        loss_dict['G_plus_C_loss'].append(G_loss+C_loss)
        loss_dict['l2_regularization'].append(l2_regularization)
        loss_dict['l1_regularization'].append(l1_regularization)
        
        # record matrix norms
        loss_dict['W_norm'].append(np.linalg.norm(W))
        if G_loss_type is not None:
            loss_dict['H_G_norm'].append(np.linalg.norm(H_G))
            if Z is not None:
                loss_dict['U_G_norm'].append(np.linalg.norm(U_G))
        if C_loss_type is not None:
            loss_dict['H_C_norm'].append(np.linalg.norm(H_C))
            if Z is not None:
                loss_dict['U_C_norm'].append(np.linalg.norm(U_C))

        # record sparsity
        loss_dict['W_sparsity'].append(100*np.count_nonzero(W == 0)/ W.size)
        if G_loss_type is not None:
            loss_dict['H_G_sparsity'].append(100*np.count_nonzero(H_G == 0)/ H_G.size)
            if Z is not None:
                loss_dict['U_G_sparsity'].append(100*np.count_nonzero(U_G == 0)/ U_G.size)
        if C_loss_type is not None:
            loss_dict['H_C_sparsity'].append(100*np.count_nonzero(H_C == 0)/ H_C.size)
            if Z is not None:
                loss_dict['U_C_sparsity'].append(100*np.count_nonzero(U_C == 0)/ U_C.size)
        #assert f_cur<=f_prev*(1+0.1), f"loss increasing (by more than 10% x previous loss): {f_prev} -> {f_cur}"
        if ((f_prev - f_cur) / max(1.0, abs(f_prev)) < tol) and _ >= min_outer:
            break
        f_prev = f_cur

    if not test:
        if post_hoc_rescale is True:
            # Normalize columns of W to L2 norm and scale rows of H to resolve scaling ambiguity
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0] = 1.0
        else:
            norms = np.ones(W.shape[1], dtype=W.dtype)
        W_normed = W / norms
        factor_matrices = {"W":W_normed}
        # scale rows of H_G and H_C
        if G_loss_type is not None:
            H_G_normed = H_G * norms[np.newaxis, :]
            factor_matrices["H_G"] = H_G_normed
            if Z is not None:
                factor_matrices["U_G"] = U_G
        if C_loss_type is not None:
            H_C_normed = H_C * norms[np.newaxis, :]
            factor_matrices["H_C"] = H_C_normed
            if Z is not None:
                factor_matrices["U_C"] = U_C
    else:
            factor_matrices = {"W":W}

    return factor_matrices, loss_dict

def SCoNE_parallel(
    G,C,Z,              # true matrices
    rank, init='random',num_init=1,n_jobs=1, # init: number of initializations (will choose one with best loss as final result), should have this=1 when test=True
    alpha=0,lambda_H_G=0, lambda_H_C=0, lambda_Gloss=1,                  # regularization parameters
    G_loss_type='kl_div', C_loss_type='kl_div', # in 'kl_div',  'fro' or None (None indicates not fitting to data, i.e. if G_loss_type=None & C_loss_type='fro then C only optimization)
    max_inner=50, # options for pgd_armijo
    rho=0.1, # options for pgd_armijo (n shrink factor)
    sigma=1e-4, # options for pgd_armijo (Armijo constant)
    inner_ftol=1e-5, # options for pgd_armijo (stopping criteria for ftol)
    max_outer=200,
    min_outer=5,
    tol=1e-6,max_ls=50,post_hoc_rescale=False,
    # if test is True, W is the only factor matrix that will be optimized (for train/test split)
    test=False,
    H_G=None, H_C=None, U_G=None, U_C=None, 
    write_all_init=False, write_all_init_path='', # write all init out
    use_gpu=False
    ):
    """
    Perform Sparse Covariate-aware Non-negative Matrix Factorization (SCoNE).

    Parameters
    ----------
    G : ndarray of shape (N, M_G) or None
        Genetic data matrix. Can be set to None to exclude from decomposition.

    C : ndarray of shape (N, M_C) or None
        Clinical data matrix. Can be set to None to exclude from decomposition.

    Z : ndarray of shape (N, M_Z) or None
        Covariate matrix. Should contain an intercept. Can be set to None to exclude from decomposition.

    rank : int
       Rank of decomposition.

    init : {'random', 'nndsvd', 'nndsvda'}, default='random'
        Initialization method for the factor matrices.

    num_init : int, default=1
        Number of initializations. The solution with the lowest final
        objective value is returned.

    n_jobs : int, default=1
        Number of worker processes used to run multiple initializations in parallel.

    alpha : float, default=0
        L2 regularization parameter for W. Should be set when lambda_H_G or lambda_H_C is greater than 0.

    lambda_H_G : float, default=0
        L1 sparsity parameter for H_G.

    lambda_H_C : float, default=0
        L1 sparsity parameter for H_C.

    lambda_Gloss : float, default=1
        Weight controlling the relative contribution of the genetic and
        clinical reconstruction losses.

    G_loss_type, C_loss_type : {'kl_div', 'fro', None}, default='kl_div'
        Reconstruction loss used for each data modality. Setting a loss type
        to None excludes that modality from the optimization.

    max_outer, min_outer : int
        Maximum and minimum number of outer optimization iterations.

    tol : float
        Convergence tolerance for the outer optimization.

    max_inner, rho, sigma, inner_ftol, max_ls
        Parameters controlling the projected gradient descent line search.

    post_hoc_rescale : bool, default=False
        Whether to rescale the learned factor matrices after optimization. If test=True this is always False.

    test : bool, default=False
        If True, only the participant factor matrix W is optimized while the
        remaining factor matrices are held fixed.

    H_G, H_C, U_G, U_C : ndarray, optional
        Precomputed factor matrices used when test=True.

    write_all_init : bool, default=False
        Whether to save the results from every initialization.

    write_all_init_path : str, default=''
        Directory in which to save initialization results.

    use_gpu: bool, default=False
        If True, use CuPy for GPU-accelerated computation. If False, use NumPy on the CPU.

    Returns
    -------
    factor_matrices : dict
        Dictionary containing the learned factor matrices.

    loss_function : dict
        Dictionary containing the total objective value, individual loss
        components, factor matrix norms, sparsity measures, and the final
        cophenetic correlation coefficient.
    """

    # Use numpy-compatible cupy API for GPU computation
    if use_gpu:
        try:
            import cupy as np
        except ImportError:
            raise ImportError(
            "CuPy is required when use_gpu=True."
            )
        if G is not None: G = np.asarray(G)
        if C is not None: C = np.asarray(C)
        if Z is not None: Z = np.asarray(Z)
        if H_G is not None: H_G = np.asarray(H_G)
        if H_C is not None: H_C = np.asarray(H_C)
        if U_G is not None: U_G = np.asarray(U_G)
        if U_C is not None: U_C = np.asarray(U_C)
    else:
        import numpy as np

    kwargs = {"G":G,"C":C,"Z":Z,              # true matrices
            "rank":rank, "init":init,
            "alpha":alpha,"lambda_H_G":lambda_H_G, "lambda_H_C":lambda_H_C, "lambda_Gloss":lambda_Gloss,                  # regularization parameters
            "G_loss_type":G_loss_type, "C_loss_type":C_loss_type, # in 'kl_div',  'fro' or None (None indicates not fitting to data, i.e. if G_loss_type=None & C_loss_type='fro then C only optimization)
            "max_inner":max_inner, # options for pgd_armijo
            "rho":rho, # options for pgd_armijo (n shrink factor)
            "sigma":sigma, # options for pgd_armijo (Armijo constant)
            "inner_ftol":inner_ftol, # options for pgd_armijo (stopping criteria for ftol)
            "max_outer":max_outer,
            "min_outer":min_outer,
            "tol":tol,"max_ls":max_ls, "post_hoc_rescale":post_hoc_rescale,
            "test":test,
            "H_G":H_G, "H_C":H_C, "U_G":U_G, "U_C":U_C}

    if n_jobs==1:
        results = []
        for run in range(num_init):
            result = alternating_opt(**kwargs)

            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(alternating_opt)(
            **kwargs) for run in range(num_init))

    # FOR: transition back to numpy array if using cupy on GPU
    if use_gpu: 
        for result in results:
            result[1] = {
                key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
                for key, value in result[1].items()
            }
    
    # FOR: writing and calculating cophenetic correlation coefficient
    W_list = []
    for run, result in enumerate(results):
        if write_all_init:
            with open(f"{write_all_init_path}_{run}_loss_function.json", "w") as f: 
                json.dump(result[1], f)
            with open(f"{write_all_init_path}_{run}_factor_matrices.pkl", "wb") as f:
                pickle.dump(result[0], f)
                # Assign each sample to its strongest component
        W_list.append(result[0]['W'])
    coph_corr = calculate_ccc(W_list)
    # FOR: writing and calculating cophenetic correlation coefficient
    
    final_factor_matrices, final_loss_dict = min(
        results,
        key=lambda x: x[1]["total_loss"][-1]
    )
    final_loss_dict['coph_corr'] = coph_corr
    
    return final_factor_matrices, final_loss_dict