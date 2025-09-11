import numpy as np

from scipy.optimize import minimize
from scipy.special import gammaln
from utilities import vec2mats, mats2vec, sigmoid


def total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # define nec. elements
    G_hat = sigmoid(W@H_G.T + Z@U_G.T)
    # clipping for log purposes
    G_hat = np.clip(G_hat, 1e-7, 1 - 1e-7)
    C_hat = W@H_C.T + Z@U_C.T
    # clipping for log purposes
    C_hat = np.clip(C_hat, 1e-7, np.inf)

    E_g = np.ones(G.shape)

    # define loss
    bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
    kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabilize
    regularization = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    if kl_div_loss < 0:
        assert True == False, "KL divergence loss negative"
    if bce_loss < 0:
        assert True == False, "BCE loss negative"
    f = bce_loss + kl_div_loss + regularization 
    return f

# Per-block factories (compute precomputes once per inner solve)

# Per-block factories (compute precomputes once per inner solve)

def make_fg_W(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # Precompute terms independent of W
    U1 = Z@U_G.T
    U2 = Z@U_C.T
    C_const = np.sum(gammaln(C + 1))
    E_g = np.ones(G.shape)
    E_c = np.ones(C.shape)
    reg = lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    def fun(x):
        W = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(W@H_G.T + U1), 1e-7, 1 - 1e-7)
        # clipping for log purposes
        C_hat = np.clip(W@H_C.T + U2, 1e-7, np.inf)
        bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
        kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat) + C_const # log(C!) can help stabilize
        f = bce_loss + kl_div_loss + lambda_W/2* np.trace(W.T @ W) + reg 
        return f
    def jac(x):
        W = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(W@H_G.T + U1), 1e-7, 1 - 1e-7)
        # clipping for log purposes
        C_hat = np.clip(W@H_C.T + U2, 1e-7, np.inf)

        G_tilde = np.divide(G,G_hat)
        G_bar = np.divide(E_g-G,E_g-G_hat)
        C_tilde = np.divide(C,C_hat)

        GW = (np.multiply(np.multiply(-G_tilde + G_bar,G_hat),E_g - G_hat)@H_G 
                + (E_c - C_tilde)@H_C + lambda_W*W)
        return GW.flatten(order='F')
    return fun, jac

def make_fg_HG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # Precompute terms independent of H_G
    U1 = Z@U_G.T
    C_hat = np.clip(W@H_C.T + Z@U_C.T, 1e-7, np.inf)
    kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabilize
    E_g = np.ones(G.shape)
    reg = lambda_W/2* np.trace(W.T @ W) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    def fun(x):
        H_G = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(W@H_G.T + U1), 1e-7, 1 - 1e-7)
        bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
        f = bce_loss + kl_div_loss + reg +  lambda_H_G/2* np.trace(H_G.T @ H_G)
        return f
    def jac(x):
        H_G = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(W@H_G.T + U1), 1e-7, 1 - 1e-7)

        G_tilde = np.divide(G,G_hat)
        G_bar = np.divide(E_g-G,E_g-G_hat)

        GH_G = (W.T@np.multiply(np.multiply(-G_tilde + G_bar,G_hat),E_g - G_hat)).T + lambda_H_G*H_G
        return GH_G.flatten(order='F')
    return fun, jac

def make_fg_HC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # Precompute terms independent of H_C
    U1 = Z@U_C.T
    E_c = np.ones(C.shape)
    C_const = np.sum(gammaln(C + 1))
    G_hat = np.clip(sigmoid(W@H_G.T + Z@U_G.T), 1e-7, 1 - 1e-7)
    E_g = np.ones(G.shape)
    bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
    reg = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G)
    def fun(x):
        H_C = x.reshape(shape, order='F')
        C_hat = np.clip(W@H_C.T + U1, 1e-7, np.inf)
        kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat) + C_const # log(C!) can help stabilize
        f = bce_loss + kl_div_loss + reg +  lambda_H_C/2* np.trace(H_C.T @ H_C)
        return f
    def jac(x):
        H_C = x.reshape(shape, order='F')
        C_hat = np.clip(W@H_C.T + U1, 1e-7, np.inf)

        C_tilde = np.divide(C,C_hat)

        GH_C = (W.T@(E_c - C_tilde)).T + lambda_H_C*H_C
        return GH_C.flatten(order='F')
    return fun, jac

def make_fg_UG(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # Precompute terms independent of U_G
    U1 = W@H_G.T
    C_hat = np.clip(W@H_C.T + Z@U_C.T, 1e-7, np.inf)
    kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabilize
    E_g = np.ones(G.shape)
    regularization = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    def fun(x):
        U_G = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(U1 + Z@U_G.T), 1e-7, 1 - 1e-7)
        bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
        f = bce_loss + kl_div_loss + regularization
        return f
    def jac(x):
        U_G = x.reshape(shape, order='F')
        # clipping for log purposes
        G_hat = np.clip(sigmoid(U1 + Z@U_G.T), 1e-7, 1 - 1e-7)

        G_tilde = np.divide(G,G_hat)
        G_bar = np.divide(E_g-G,E_g-G_hat)

        GU_G = (Z.T@(np.multiply(np.multiply(-G_tilde + G_bar,G_hat),E_g - G_hat))).T
        return GU_G.flatten(order='F')
    return fun, jac

def make_fg_UC(shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C):
    # Precompute terms independent of U_C
    U1 = W@H_C.T
    E_c = np.ones(C.shape)
    C_const = np.sum(gammaln(C + 1))
    G_hat = np.clip(sigmoid(W@H_G.T + Z@U_G.T), 1e-7, 1 - 1e-7)
    E_g = np.ones(G.shape)
    bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
    regularization = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    def fun(x):
        U_C = x.reshape(shape, order='F')
        C_hat = np.clip(U1 + Z@U_C.T, 1e-7, np.inf)
        kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat) + C_const
        f = bce_loss + kl_div_loss + regularization
        return f
    def jac(x):
        U_C = x.reshape(shape, order='F')
        C_hat = np.clip(U1 + Z@U_C.T, 1e-7, np.inf)

        C_tilde = np.divide(C,C_hat)

        GU_C = (Z.T@(E_c - C_tilde)).T
        return GU_C.flatten(order='F')
    return fun, jac

# speed up when running separate function for optimization of each matrix (unlike in JointMF_AAO)
def alternating_opt(
    G,C,Z,              # true matrices
    W, H_G, H_C, U_G, U_C,  # initializations
    lambda_W, lambda_H_G, lambda_H_C,                   # regularization parameters
    method,                 # method and options for scipy minimize
    options,
    max_outer=30,
    tol=1e-4,
    nonneg=True             # set per-block L-BFGS-B bounds to [0, +inf)
):
    def one_block_update(name, X, method, options):
        x0 = X.flatten(order='F') 
        bnds = [(0, None)] * x0.size if nonneg else None

        if name == "W":
            fun, jac = make_fg_W(W.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        elif name == "H_G":
            fun, jac = make_fg_HG(H_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        elif name == "H_C":
            fun, jac = make_fg_HC(H_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        elif name == "U_G":
            fun, jac = make_fg_UG(U_G.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        elif name == "U_C":
            fun, jac = make_fg_UC(U_C.shape, G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        else:
            raise ValueError(f"Unknown block {name}")

        res = minimize(fun, x0, method=method, jac=jac, bounds=bnds, options=options)
        return res.x.reshape(X.shape, order='F')

    # initial objective
    f_prev = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)

    for _ in range(max_outer):
        W   = one_block_update("W",   W, method, options)
        H_G = one_block_update("H_G", H_G, method, options)
        H_C = one_block_update("H_C", H_C, method, options)
        U_G = one_block_update("U_G", U_G, method, options)
        U_C = one_block_update("U_C", U_C, method, options)

        f_cur = total_loss(G, C, Z, W, H_G, H_C, U_G, U_C, lambda_W, lambda_H_G, lambda_H_C)
        if (f_prev - f_cur) / max(1.0, abs(f_prev)) < tol:
            break
        f_prev = f_cur

    return W, H_G, H_C, U_G, U_C
