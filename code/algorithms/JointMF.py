# Joint matrix factorization, loss function looks like: (min W,H_G,U_G,H_C,U_C) 1/2||G-WH_G-ZU_G||_{BCE Loss} + 1/2||C-WH_C-ZU_C||_{Generalized KL divergence}


import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import math
from functools import partial
from scipy.optimize import minimize
from scipy.special import gammaln
from utilities import vec2mats, mats2vec

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# def soft_plus(x):
#   return np.log(1 + np.exp(x))

def JMF_FG(v,G,C,Z,rank,lambda_W,lambda_H_G,lambda_H_C):
    '''
    G,C,Z: known
    '''
    W,H_G,H_C,U_G,U_C = vec2mats(v, shapes=[(G.shape[0],rank),(G.shape[1],rank),(C.shape[1],rank),(G.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])

    # define nec. elements
    G_hat = sigmoid(W@H_G.T + Z@U_G.T)
    # clipping for log purposes
    G_hat = np.clip(G_hat, 1e-7, 1 - 1e-7)
    C_hat = W@H_C.T + Z@U_C.T
    # clipping for log purposes
    C_hat = np.clip(C_hat, 1e-7, 1 - 1e-7)

    E_g = np.ones(G.shape)
    E_c = np.ones(C.shape)

    G_tilde = np.divide(G,G_hat)
    G_bar = np.divide(E_g-G,E_g-G_hat)
    C_tilde = np.divide(C,C_hat)

    # define loss
    bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
    kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabilize
    regularization = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    if kl_div_loss < 0:
        assert True == False, "KL divergence loss negative"
    if bce_loss < 0:
        assert True == False, "BCE loss negative"
    f = bce_loss + kl_div_loss + regularization 

    # calculate gradients wrt all unknowns
    # precompute certain factors
    U1 = np.multiply(np.multiply(-G_tilde + G_bar,G_hat),E_g - G_hat)
    U2 = E_c - C_tilde

    GW = U1@H_G + U2@H_C + lambda_W*W
    GH_G = (W.T@U1).T + lambda_H_G*H_G
    GH_C = (W.T@U2).T + lambda_H_C*H_C
    GU_G = (Z.T@U1).T
    GU_C = (Z.T@U2).T

    # Convert gradients to vector form
    g = mats2vec([GW, GH_G, GH_C, GU_G, GU_C])
    return f, g

def loss_calc(v,G,C,Z,rank,lambda_W,lambda_H_G,lambda_H_C): # TODO: maybe timing takes longer because we calculate loss twice
    W, H_G, H_C, U_G, U_C = vec2mats(v, shapes=[(G.shape[0],rank),(G.shape[1],rank),(C.shape[1],rank),(G.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])
    # define nec. elements
    G_hat = sigmoid(W@H_G.T + Z@U_G.T)
    # clipping for log purposes
    G_hat = np.clip(G_hat, 1e-7, 1 - 1e-7)
    C_hat = W@H_C.T + Z@U_C.T
    C_hat = np.clip(C_hat, 1e-7, 1 - 1e-7)

    E_g = np.ones(G.shape)

    bce_loss = np.sum(-np.multiply(G,np.log(G_hat)) - np.multiply(E_g-G,np.log(E_g-G_hat)))
    kl_div_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabilize
    regularization = lambda_W/2* np.trace(W.T @ W) +  lambda_H_G/2* np.trace(H_G.T @ H_G) +  lambda_H_C/2* np.trace(H_C.T @ H_C)
    f = bce_loss + kl_div_loss + regularization 
    return [f, bce_loss, kl_div_loss, regularization]


def jmf(G, C, Z, rank, lambda_W, lambda_H_G, lambda_H_C, method, writer, options, W=None, H_G=None, H_C=None, U_G=None, U_C=None):
    '''
    G: genetic data matrix
    C: clinical data matrix
    rank: specified rank for decomposition
    lambda_W,lambda_H_G,lambda_H_C: regularization paramaters for W, H_G, H_C
    method: solver method from scipy.optimize.minimize
    writer: tensorboard writer if defined (else None)
    make sure options are specific to solver method specified (if unsure look at scipy.optimize.minimize documentation)
    W, H_G, H_C, U_G, U_C exist for initializing model (if any of them not defined, model is initialized within run)
    '''
    # make sure options are specific to solver method specified (if unsure look at scipy.optimize.minimize documentation)
    loss_history = []
    def callback(v,G,C,Z,rank):
        loss_history.append(loss_calc(v,G,C,Z,rank,lambda_W,lambda_H_G,lambda_H_C))
    callback_with_args = partial(callback, G=G, C=C, Z=Z, rank=rank)
    # tol=0 forces specific number of iterations
    if not all(matrix is not None for matrix in [W, H_G, H_C, U_G, U_C]): # at least one matrix not pre-initialized - initialize all
        W = np.random.random((G.shape[0], rank))
        H_G = np.random.random((G.shape[1], rank))
        H_C = np.random.random((C.shape[1], rank))
        U_G = np.random.random((G.shape[1], Z.shape[1]))
        U_C = np.random.random((C.shape[1], Z.shape[1]))
    v_init = mats2vec([W, H_G, H_C, U_G, U_C])
    bounds = [(0.0, None)] * v_init.size # positive projection
    result = minimize(JMF_FG,v_init, method=method, jac=True, bounds=bounds, args=(G,C,Z,rank,lambda_W,lambda_H_G,lambda_H_C),options=options,callback=callback_with_args)
    W,H_G,H_C,U_G,U_C= vec2mats(result.x, shapes=[(G.shape[0],rank),(G.shape[1],rank),(C.shape[1],rank),(G.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])
    # add in loss
    if writer is not None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.tensorboard import SummaryWriter
        for iteration in range(len(loss_history)):
            writer.add_scalar('Loss', loss_history[iteration][0], iteration)
            writer.add_scalar('BCE Loss', loss_history[iteration][1], iteration)
            writer.add_scalar('Poisson Loss', loss_history[iteration][2], iteration)
    return W, H_G, H_C, U_G, U_C, loss_history, result.success, result.message
