# Joint matrix factorization, loss function looks like: (min A,B,D) 1/2||X-AB||^2 + 1/2||C-AD||^2

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

def soft_plus(x):
  return np.log(1 + np.exp(x))

def JMF_FG(v,X,C,Z,rank):
    '''
    X,C,Z: known
    this performs all at once optimization
    '''
    A,B,D,B_prime,D_prime = vec2mats(v, shapes=[(X.shape[0],rank),(X.shape[1],rank),(C.shape[1],rank),(X.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])

    # define nec. elements
    X_hat = sigmoid(A@B.T + Z@B_prime.T)
    # clipping for log purposes
    X_hat = np.clip(X_hat, 1e-7, 1 - 1e-7)
    C_hat = soft_plus(A@D.T + Z@D_prime.T)

    E_x = np.ones(X.shape)
    E_c = np.ones(C.shape)

    X_tilde = np.divide(X,X_hat)
    X_bar = np.divide(E_x-X,E_x-X_hat)
    C_tilde = np.divide(C,C_hat)

    # define loss
    bce_loss = np.sum(-np.multiply(X,np.log(X_hat)) - np.multiply(E_x-X,np.log(E_x-X_hat)))
    poisson_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabalize
    if poisson_loss < 0:
        assert True == False, "Poisson loss negative"
    if bce_loss < 0:
        assert True == False, "BCE loss negative"
    f = bce_loss + poisson_loss

    # calculate gradients wrt all unknowns
    # precompute certain factors
    U1 = np.multiply(np.multiply(-X_tilde + X_bar,X_hat),E_x - X_hat)
    U2 = E_c - C_tilde
    U3 = sigmoid(A@D.T + Z@D_prime.T) # derivative of softplus is sigmoid
    U2_U3 = np.multiply(U2,U3)

    GA = U1@B + U2_U3@D
    GB = (A.T@U1).T
    GD = (A.T@U2_U3).T
    GB_prime = (Z.T@U1).T
    GD_prime = (Z.T@U2_U3).T

    # Convert gradients to vector form
    g = mats2vec([GA, GB, GD, GB_prime, GD_prime])
    return f, g

def loss_calc(v,X,C,Z,rank):
    A, B, D, B_prime, D_prime = vec2mats(v, shapes=[(X.shape[0],rank),(X.shape[1],rank),(C.shape[1],rank),(X.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])
    X_hat = sigmoid(A@B.T + Z@B_prime.T)
    # clipping for log purposes
    X_hat = np.clip(X_hat, 1e-7, 1 - 1e-7)
    C_hat = soft_plus(A@D.T + Z@D_prime.T)
    E_x = np.ones(X.shape)

    # define loss
    bce_loss = np.sum(-np.multiply(X,np.log(X_hat)) - np.multiply(E_x-X,np.log(E_x-X_hat)))
    poisson_loss = np.sum(-np.multiply(C,np.log(C_hat)) + C_hat + gammaln(C + 1)) # log(C!) can help stabalize
    # print(f'bce loss: {bce_loss}')
    # print(f'poisson loss:{poisson_loss}')
    f = bce_loss + poisson_loss
    return [f, bce_loss, poisson_loss]


def jmf(X, C, Z, rank, A, B, D, B_prime, D_prime,  method, writer, options):
    # make sure options are specific to solver method specified (if unsure look at scipy.optimize.minimize documentation)
    loss_history = []
    def callback(v,X,C,Z,rank):
        loss_history.append(loss_calc(v,X,C,Z,rank))
    callback_with_args = partial(callback, X=X, C=C, Z=Z, rank=rank)
    # tol=0 forces specific number of iterations
    v_init = mats2vec([A, B, D, B_prime, D_prime])
    result = minimize(JMF_FG,v_init, method=method, jac=True, args=(X,C,Z,rank),options=options,callback=callback_with_args)
    A,B,D, B_prime, D_prime= vec2mats(result.x, shapes=[(X.shape[0],rank),(X.shape[1],rank),(C.shape[1],rank),(X.shape[1],Z.shape[1]),(C.shape[1],Z.shape[1])])
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
    return A,B,D,B_prime, D_prime, loss_history, result.success, result.message

