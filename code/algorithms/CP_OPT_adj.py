# CP OPT with adjustment for covariates Z
import numpy as np
from utilities import f_unfold, vec2mats, mats2vec
import tensorly as tl
from tensortools.operations import khatri_rao
from functools import partial
from scipy.optimize import minimize


def OPT_FG(v, T, Z, rank, reg_param): # TODO: FIX
    """Computes the CP function and gradient for a 3-way tensor."""
    # Convert vector v to matrices A, B, D
    A, B, D, B_prime, D_prime = vec2mats(v, shapes=[(T.shape[0],rank), (T.shape[1],rank), (T.shape[2],rank), (T.shape[1],Z.shape[1]), (T.shape[2],Z.shape[1])])
    
    # Compute Gramian matrices
    S1 = A.T @ A
    S2 = B.T @ B
    S3 = D.T @ D
    S4 = Z.T @ Z
    S5 = B_prime.T @ B_prime
    S6 = D_prime.T @ D_prime
    
    # compute other intermediates
    V1 = Z.T@A
    V2 = B_prime.T@B
    V3 = D_prime.T@D

    # Compute gradients for A, B, C
    G1 = A @ (np.multiply(S3, S2)) - (f_unfold(T,0) @ khatri_rao([D, B]) - Z@np.multiply(V3, V2))
    G2 = B @ (np.multiply(S3, S1)+ 2*reg_param*(B.T@B-np.identity(rank))) - (f_unfold(T,1) @ khatri_rao([D, A]) - B_prime@np.multiply(V3,V1))
    G3 = D @ (np.multiply(S2,S1)+ 2*reg_param*(D.T@D-np.identity(rank))) - (f_unfold(T,2) @ khatri_rao([B,A]) - D_prime@np.multiply(V2,V1))
    # Z is fixed
    G5 = B_prime @ (np.multiply(S6, S4)) - (f_unfold(T,1) @ khatri_rao([D_prime, Z]) - B@np.multiply(V3.T,V1.T))
    G6 = D_prime @ (np.multiply(S5,S4)) - (f_unfold(T,2) @ khatri_rao([B_prime,Z]) - D@np.multiply(V2.T,V1.T))
    
    # Compute function value
    LS = (1/2) * np.linalg.norm(T - tl.cp_to_tensor((np.array([1]*rank),[A,B,D])) - tl.cp_to_tensor((np.array([1]*Z.shape[1]),[Z,B_prime,D_prime])))**2
    L_ortho = (reg_param/2) * np.linalg.norm(S2 - np.identity(rank))**2 + (reg_param/2) * np.linalg.norm(S3 - np.identity(rank))**2 # add in ortho loss
    f = LS + L_ortho

    # Convert gradients to vector form
    g = mats2vec([G1, G2, G3, G5, G6])
    
    return f, g

def rel_error_calc(v,T,Z,T_norm,rank, reg_param):
    A, B, D, B_prime, D_prime = vec2mats(v, shapes=[(T.shape[0],rank), (T.shape[1],rank), (T.shape[2],rank), (T.shape[1],Z.shape[1]), (T.shape[2],Z.shape[1])])
    # relative error calc
    rel_error = np.linalg.norm(T - tl.cp_to_tensor((np.array([1]*rank),[A,B,D])) - tl.cp_to_tensor((np.array([1]*Z.shape[1]),[Z,B_prime,D_prime])))/T_norm
    # LS (Least squares)
    LS = (1/2) * np.linalg.norm(T - tl.cp_to_tensor((np.array([1]*rank),[A,B,D])) - tl.cp_to_tensor((np.array([1]*Z.shape[1]),[Z,B_prime,D_prime])))**2
    # L_ortho (orthogonal loss)
    L_ortho =  (reg_param/2) * np.linalg.norm(B.T@B - np.identity(rank))**2 + (reg_param/2) * np.linalg.norm(D.T@D - np.identity(rank))**2 # add in ortho loss
    return [rel_error, LS, L_ortho]


# Callback function to store loss at each step
def cp_opt_adjusted(T, Z, T_norm, rank, a, b, d, b_prime, d_prime, reg_param, method, writer, options):
    # make sure options are specific to solver method specified (if unsure look at scipy.optimize.minimize documentation)
    loss_history = []
    def callback(v, T, Z, T_norm, rank, reg_param):
        loss_history.append(rel_error_calc(v, T, Z, T_norm,rank, reg_param))
    callback_with_args = partial(callback, T=T, Z=Z, T_norm=T_norm,rank=rank, reg_param=reg_param)
    # tol=0 forces specific number of iterations
    v_init = mats2vec([a,b,d,b_prime,d_prime])
    result = minimize(OPT_FG,v_init, method=method, jac=True, args=(T, Z, rank, reg_param),options=options,callback=callback_with_args)
    if result.success is not True:
        print(f'OPTIMIZATION NOT SUCCESSFUL: {result}', flush=True)
    A,B,D,B_prime,D_prime = vec2mats(result.x, shapes=[(T.shape[0],rank), (T.shape[1],rank), (T.shape[2],rank), (T.shape[1],Z.shape[1]), (T.shape[2],Z.shape[1])])
    # add in loss
    if writer is not None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.tensorboard import SummaryWriter
        for iteration in range(len(loss_history)):
            writer.add_scalar('Relative Error', loss_history[iteration][0], iteration)
            writer.add_scalar('LS', loss_history[iteration][1], iteration)
            writer.add_scalar('L_ortho', loss_history[iteration][2], iteration)
    return A,B,D,B_prime, D_prime, loss_history, result.success, result.message