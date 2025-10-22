import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.special import gammaln
from utilities import vec2mats, mats2vec, sigmoid

def compute_loss(X,X_hat,loss_type):
    if loss_type == 'kl_div':
        return np.sum(-np.multiply(X,np.log(X_hat)) + X_hat + gammaln(X + 1)) # log(X!) can help stabilize/make pos.
    elif loss_type == 'bce':
        E_x = np.ones(X.shape)
        return np.sum(-np.multiply(X,np.log(X_hat)) - np.multiply(E_x-X,np.log(E_x-X_hat)),dtype=np.float64)
    elif loss_type == 'fro':
        return (1/2)*np.dot((X - X_hat).ravel(), (X - X_hat).ravel())
    elif loss_type is None:
        return 0
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'bce', 'fro'"

def l1_norm(x):
    return np.sum(np.abs(x),dtype=np.float64)

def compute_jac(sample_matrix, X, X_hat, loss_type, for_W=False):
    '''
    This function works to compute the Jacobian (unflattened) given loss type in  'kl_div', 'bce', 'fro'
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
    elif loss_type == 'bce':
        E_x = np.ones(X.shape)
        X_tilde = np.divide(X,X_hat)
        X_bar = np.divide(E_x-X,E_x-X_hat)
        if for_W:
            GX = np.multiply(np.multiply(-X_tilde + X_bar,X_hat),E_x - X_hat)@sample_matrix
        else:
            GX = (sample_matrix.T@np.multiply(np.multiply(-X_tilde + X_bar,X_hat),E_x - X_hat)).T
    elif loss_type == 'fro':
        if for_W:
            GX = (X_hat-X)@sample_matrix
        else:
            GX = (sample_matrix.T@(X_hat-X)).T
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'bce', 'fro'"
    return GX

def get_X_hat(W,H,Z,U,loss_type):
    X_hat = W@H.T + Z@U.T
    if loss_type == 'bce':
        X_hat = sigmoid(X_hat)
        X_hat = np.clip(X_hat, 1e-7, 1 - 1e-7) # clipping for log purposes
    elif loss_type in ['kl_div','fro']:
        X_hat = np.clip(X_hat, 1e-7, np.inf) # clipping for log purposes
    elif loss_type is None:
        X_hat = None
    else: assert True == False, f"{loss_type} not a valid loss type, should be one of 'kl_div', 'bce', 'fro'"
    return X_hat

def total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # define nec. elements
    if G_loss_type is not None:
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        G_loss = compute_loss(G,G_hat,loss_type=G_loss_type)
    else: G_loss = 0
    if C_loss_type is not None:
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        C_loss = compute_loss(C,C_hat,loss_type=C_loss_type)
    else: C_loss = 0

    regularization = lambda_W*l1_norm(W) +  lambda_H_G*l1_norm(H_G) +  lambda_H_C*l1_norm(H_C) # l1 reg.

    if G_loss < 0: assert True == False, f"G_loss with loss type {G_loss_type} negative"
    if C_loss < 0: assert True == False, f"C_loss with loss type {C_loss_type} negative"
    loss = lambda_Gloss*G_loss + C_loss + regularization
    return loss, lambda_Gloss*G_loss, C_loss, regularization


def make_fg_W(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of W
    reg = lambda_H_G*l1_norm(H_G)+ lambda_H_C*l1_norm(H_C)
    def _forwardG(x):
        W = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        return G_hat
    def _forwardC(x):
        W = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
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
        loss = lambda_Gloss*G_loss + C_loss + lambda_W*l1_norm(W) + reg 
        return loss

    def g(x):
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


        GW = (lambda_Gloss*GW_wrt_G + GW_wrt_C)
        return GW.flatten(order='F')

    return f,g

def make_fg_HG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of H_G
    C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
    C_loss = compute_loss(C,C_hat,C_loss_type)
    reg = lambda_W*l1_norm(W) +  lambda_H_C*l1_norm(H_C)
    def _forward(x):
        H_G = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type) # could cut down on matrix multiplications if use precalculated Z@U_G.T for fun and jac
        return G_hat
    def f(x):
        G_hat = _forward(x)
        G_loss = compute_loss(G,G_hat,G_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss + reg + lambda_H_G*l1_norm(H_G)
        return loss
    def g(x):
        G_hat = _forward(x)
        # compute jac
        GH_G = lambda_Gloss*compute_jac(W, G, G_hat, G_loss_type)
        return GH_G.flatten(order='F')

    return f,g

def make_fg_HC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of H_C
    G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
    G_loss = compute_loss(G,G_hat,G_loss_type)
    reg = lambda_W*l1_norm(W) +  lambda_H_G*l1_norm(H_G)
    def _forward(x):
        H_C = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
        C_hat = _forward(x)
        C_loss = compute_loss(C,C_hat,C_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss + reg +  lambda_H_C*l1_norm(H_C)
        return loss
    def g(x):
        C_hat = _forward(x)
        # compute jac
        GH_C = compute_jac(W, C, C_hat, C_loss_type)
        return GH_C.flatten(order='F')

    return f,g

def make_fg_UG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of U_G
    C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
    C_loss = compute_loss(C,C_hat,C_loss_type)
    regularization = lambda_W*l1_norm(W) +  lambda_H_G*l1_norm(H_G) +  lambda_H_C*l1_norm(H_C)
    def _forward(x):
        U_G = x.reshape(shape, order='F')
        G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
        return G_hat
    def f(x):
        G_hat = _forward(x)
        G_loss =  compute_loss(G,G_hat,G_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss + regularization
        return loss
    def g(x):
        G_hat = _forward(x)
        # compute jac
        GU_G = lambda_Gloss*compute_jac(Z, G, G_hat, G_loss_type)
        return GU_G.flatten(order='F')

    return f,g

def make_fg_UC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type):
    # Precompute terms independent of U_C
    G_hat = get_X_hat(W,H_G,Z,U_G,G_loss_type)
    G_loss = compute_loss(G,G_hat,G_loss_type)
    regularization = lambda_W*l1_norm(W) +  lambda_H_G*l1_norm(H_G) +  lambda_H_C*l1_norm(H_C)
    def _forward(x):
        U_C = x.reshape(shape, order='F')
        C_hat = get_X_hat(W,H_C,Z,U_C,C_loss_type)
        return C_hat
    def f(x):
        C_hat = _forward(x)
        C_loss = compute_loss(C,C_hat,C_loss_type)
        # compute fun
        loss = lambda_Gloss*G_loss + C_loss + regularization
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

def pgd_armijo(fun, grad, x0, max_iter=500, rho=0.1, sigma=1e-4, ftol=1e-12, gtol=1e-8, l1=0):
    """
    Projected gradient descent with Armijo rule.

    f: objective function, g: gradient
    sigma (Armijo constant in (0,1)); rho (shrink factor in (0,1))
    l1: lambda for L1 soft-thresholding sparsity parameter
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

        if np.linalg.norm(g)<gtol: # early stopping if gradient norm is small
            break

        x_new, s = x_and_s(n, x, g)
        f_new = fun(x_new)

        if f_new - f <= sigma * np.vdot(g, s):
            # grow n: n <- n / rho until Armijo fails or projection makes no change
            while True:
                n_next = n / rho
                x_next, s_next = x_and_s(n_next, x, g)
                if np.allclose(x_next, x_new):
                    break
                f_next = fun(x_next)
                if not (f_next - f <= sigma * np.vdot(g, s_next)):
                    break
                n, x_new, s, f_new = n_next, x_next, s_next, f_next
        else:
            while True:
                n_next= n * rho
                x_next, s_next = x_and_s(n_next, x, g)
                f_next = fun(x_next)
                if f_next - f <= sigma * np.vdot(g, s_next):
                    n, x_new, s, f_new = n_next, x_next, s_next, f_next
                    break
                n, x_new, s, f_new = n_next, x_next, s_next, f_next
        
        # implement early stopping (with gtol and ftol)
        if abs(f_new - f)/max(1.0, abs(f)) < ftol:
            return x_new

        x = x_new

    return x

def alternating_opt(
    G,C,Z,              # true matrices
    rank, num_init, # init: number of initializations (will choose one with best loss as final result)
    lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss,                  # regularization parameters
    G_loss_type, C_loss_type, # in 'kl_div', 'bce', 'fro' or None (None indicates not fitting to data, i.e. if G_loss_type=None & C_loss_type='fro then C only optimization)
    max_inner=50, # options for pgd_armijo
    rho=0.1, # options for pgd_armijo (n shrink factor)
    sigma=1e-4, # options for pgd_armijo (Armijo constant)
    inner_ftol=1e-5, # options for pgd_armijo (stopping criteria for ftol)
    inner_gtol=1e-8,
    max_outer=30,
    min_outer=5,
    tol=1e-6
):
    '''
    To remove sparsity (CoNE) - set all lambda = 0
    To remove covariates Z (HNMF, SHNMF) - set Z = np.zeros((Z.shape[0],Z.shape[1]))
    To run G only (GNMF)
    To run C only (CNMF)
    To run with all Frobenius norm (SCoNE (Fro))
    '''
    def one_block_update(name, X):
        x0 = X.flatten(order='F') 

        if name == "W":
            f,g = make_fg_W(W.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = lambda_W
        elif name == "H_G":
            f,g = make_fg_HG(H_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = lambda_H_G
        elif name == "H_C":
            f,g = make_fg_HC(H_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = lambda_H_C
        elif name == "U_G":
            f,g = make_fg_UG(U_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = 0
        elif name == "U_C":
            f,g = make_fg_UC(U_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            l1 = 0
        else:
            raise ValueError(f"Unknown block {name}")
        x = pgd_armijo(f, g, x0, max_iter=max_inner, rho=rho, sigma=sigma, ftol=inner_ftol, gtol=inner_gtol, l1=l1) # TODO: change f to smooth part only
        #res = minimize(fun, x0, method=method, jac=True, bounds=bnds, options=options)
        return x.reshape(X.shape, order='F')

    best_total_loss = np.inf
    for run in range(num_init):
        # initialize factor matrices
        W = np.random.uniform(low=0.1,high=1,size=(G.shape[0], rank))
        H_G = np.random.uniform(low=0.1,high=1,size=(G.shape[1], rank))
        H_C = np.random.uniform(low=0.1,high=1,size=(C.shape[1], rank)) 
        U_G = np.random.uniform(low=0.1,high=1,size=(G.shape[1], Z.shape[1]))
        U_C = np.random.uniform(low=0.1,high=1,size=(C.shape[1], Z.shape[1]))

        # initial objective
        loss_dict = defaultdict(list)
        f_prev, G_loss, C_loss, regularization = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)

        for _ in range(max_outer):
            W   = one_block_update("W", W)
            if G_loss_type is not None:
                H_G = one_block_update("H_G", H_G)
                U_G = one_block_update("U_G", U_G)
            if C_loss_type is not None:
                H_C = one_block_update("H_C", H_C)
                U_C = one_block_update("U_C", U_C)

            f_cur, G_loss, C_loss, regularization = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C, lambda_Gloss, G_loss_type, C_loss_type)
            loss_dict['total_loss'].append(f_cur)
            loss_dict['G_loss'].append(G_loss)
            loss_dict['C_loss'].append(C_loss)
            loss_dict['G_plus_C_loss'].append(G_loss+C_loss)
            loss_dict['regularization'].append(regularization)
            
            # record matrix norms
            loss_dict['W_norm'].append(np.linalg.norm(W))
            loss_dict['H_G_norm'].append(np.linalg.norm(H_G))
            loss_dict['H_C_norm'].append(np.linalg.norm(H_C))
            loss_dict['U_G_norm'].append(np.linalg.norm(U_G))
            loss_dict['U_C_norm'].append(np.linalg.norm(U_C))

            # record sparsity
            loss_dict['W_sparsity'].append(100*np.count_nonzero(W == 0)/ W.size)
            loss_dict['H_G_sparsity'].append(100*np.count_nonzero(H_G == 0)/ H_G.size)
            loss_dict['H_C_sparsity'].append(100*np.count_nonzero(H_C == 0)/ H_C.size)
            loss_dict['U_G_sparsity'].append(100*np.count_nonzero(U_G == 0)/ U_G.size) # expect to stay fairly constant over time
            loss_dict['U_C_sparsity'].append(100*np.count_nonzero(U_C == 0)/ U_C.size) # expect to stay fairly constant over time
            print(f'G loss: {G_loss}',flush=True)
            print(f'C loss: {C_loss}',flush=True)
            print(f'regularization: {regularization}',flush=True)
            assert f_prev - f_cur>=0, f"loss increasing: {f_prev} -> {f_cur}"
            if ((f_prev - f_cur) / max(1.0, abs(f_prev)) < tol) and _ >= min_outer:
                break
            f_prev = f_cur

        if f_cur < best_total_loss:
            best_total_loss = f_cur # reset
            final_loss_dict = loss_dict
            # Normalize columns of W to L2 norm and scale rows of H to resolve scaling ambiguity
            norms = np.linalg.norm(W, axis=0)
            norms[norms == 0] = 1.0
            W_normed = W / norms
            # scale rows of H_G and H_C
            H_G_normed = H_G * norms[np.newaxis, :]
            H_C_normed = H_C * norms[np.newaxis, :]
            final_factor_matrices = {"W":W_normed, "H_G":H_G_normed, "H_C":H_C_normed, "U_G":U_G, "U_C":U_C}
    return final_factor_matrices, final_loss_dict
