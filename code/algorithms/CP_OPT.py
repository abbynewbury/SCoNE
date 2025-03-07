import numpy as np
from utilities import f_unfold, vec2mats, mats2vec
import tensorly as tl
from tensortools.operations import khatri_rao
from functools import partial
from scipy.optimize import minimize


def OPT_FG(v, T, T_norm_squared, rank, reg_param):
    """Computes the CP function and gradient for a 3-way tensor."""
    # Convert vector v to matrices A, B, D
    A, B, D = vec2mats(v, rank, T.shape[0], T.shape[1], T.shape[2])
    
    # Compute Gramian matrices
    S1 = A.T @ A
    S2 = B.T @ B
    S3 = D.T @ D
    
    # Compute gradients for A, B, C
    G1 = A @ (np.multiply(S3, S2)) - f_unfold(T,0) @ khatri_rao([D, B])
    G2 = B @ (np.multiply(S3, S1)+ 2*reg_param*(B.T@B-np.identity(rank))) - f_unfold(T,1) @ khatri_rao([D, A])
    
    # Save intermediate calculations
    V3 = np.multiply(S2,S1)
    U3 = f_unfold(T,2) @ khatri_rao([B,A])
    G3 = D @ (V3+ 2*reg_param*(D.T@D-np.identity(rank))) - U3
    
    # Compute function value
    LS = (1/2) * T_norm_squared - np.dot(D.flatten(order='F'),U3.flatten(order='F')) + (1/2) * np.dot(V3.flatten(order='F'), S3.flatten(order='F'))
    L_ortho =  (reg_param/2) * np.linalg.norm(S2 - np.identity(rank))**2 + (reg_param/2) * np.linalg.norm(S3 - np.identity(rank))**2 # add in ortho loss
    f = LS+ L_ortho

    # Convert gradients to vector form
    g = mats2vec(G1, G2, G3)
    
    return f, g

def rel_error_calc(v,T,T_norm,rank, reg_param):
    A, B, D = vec2mats(v, rank, T.shape[0], T.shape[1], T.shape[2])
    # relative error calc
    rel_error = np.linalg.norm(T - tl.cp_to_tensor((np.array([1]*rank),[A,B,D])))/T_norm
    # LS (Least squares)
    LS = (1/2) * np.linalg.norm(T - tl.cp_to_tensor((np.array([1]*rank),[A,B,D])))**2
    # L_ortho (orthogonal loss)
    L_ortho =  (reg_param/2) * np.linalg.norm(B.T@B - np.identity(rank))**2 + (reg_param/2) * np.linalg.norm(D.T@D - np.identity(rank))**2# add in ortho loss
    return [rel_error, LS, L_ortho]


# Callback function to store loss at each step
def cp_opt(T, T_norm, rank, a, b, d, reg_param, method, options):
    # make sure options are specific to solver method specified (if unsure look at scipy.optimize.minimize documentation)
    loss_history = []
    def callback(v, T,T_norm,rank, reg_param):
        loss_history.append(rel_error_calc(v, T, T_norm,rank, reg_param))
    callback_with_args = partial(callback, T=T, T_norm=T_norm,rank=rank, reg_param=reg_param)
    # tol=0 forces specific number of iterations
    v_init = mats2vec(a,b,d)
    result = minimize(OPT_FG,v_init, method=method, jac=True, args=(T, T_norm**2, rank, reg_param),options=options,callback=callback_with_args)
    if result.success is not True:
        print(f'OPTIMIZATION NOT SUCCESSFUL: {result}', flush=True)
    A,B,D = vec2mats(result.x, rank, T.shape[0], T.shape[1], T.shape[2])
    return A,B,D,loss_history, result.success
        