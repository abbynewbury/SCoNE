# CP ALS with adjustment for covariates Z
import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import math


def cp_als_adjusted(T, Z, T_norm, rank, A, B, D, B_prime, D_prime, writer, max_iter=50, tol=1e-4):
    # similar to Algorithm 11.1 in Kolda textbook
    # writer is tensorboard writer

    S2 = B.T@B
    S3 = D.T@D
    S4 = Z.T@Z
    S5 = B_prime.T@B_prime
    S6 = D_prime.T@D_prime

    e = []
    rel_error = []

    for epoch in range(max_iter):
        # fix T'' = T - [[A,B,D]]
        T_prime_prime = T - tl.cp_to_tensor((np.array([1]*rank), [A,B,D]))
        # optimize b'
        # solving BV2=U2
        V5 = np.multiply(S6, S4)
        U5 = f_unfold(T_prime_prime,1)@khatri_rao([D_prime,Z])
        B_prime = np.linalg.solve(V5.T,U5.T).T
        S5 = B_prime.T@B_prime
        T_prime = T - tl.cp_to_tensor((np.array([1]*Z.shape[1]), [Z,B_prime,D_prime]))

        # optimize d'
        # solving DV3=U3
        V6 = np.multiply(S5,S4)
        U6 = f_unfold(T_prime_prime,2)@khatri_rao([B_prime,Z])
        D_prime = np.linalg.solve(V6.T,U6.T).T
        S6 = D_prime.T@D_prime
        T_prime = T - tl.cp_to_tensor((np.array([1]*Z.shape[1]), [Z,B_prime,D_prime]))


        # fix T' = T-[[Z,B',D']]
        T_prime = T - tl.cp_to_tensor((np.array([1]*Z.shape[1]), [Z,B_prime,D_prime]))
        # optimize a
        # solving AV1 = U1
        V1 = np.multiply(S2, S3) # no reg param here
        U1 = f_unfold(T_prime,0)@khatri_rao([D,B])
        A = np.linalg.solve(V1.T,U1.T).T
        S1 = A.T@A

        # optimize b
        # solving BV2=U2
        V2 = np.multiply(S3, S1)
        U2 = f_unfold(T_prime,1)@khatri_rao([D,A])
        B = np.linalg.solve(V2.T,U2.T).T
        S2 = B.T@B

        # optimize d
        # solving DV3=U3
        V3 = np.multiply(S2,S1)
        U3 = f_unfold(T_prime,2)@khatri_rao([B,A])
        D = np.linalg.solve(V3.T,U3.T).T
        S3 = D.T@D
        T_prime_prime = T - tl.cp_to_tensor((np.array([1]*rank), [A,B,D]))

        a_b_d = tl.cp_to_tensor((np.array([1]*rank), [A,B,D]))
        z_b_prime_d_prime = tl.cp_to_tensor((np.array([1]*Z.shape[1]), [Z,B_prime,D_prime]))
        e_t = np.linalg.norm(T - a_b_d - z_b_prime_d_prime)
        e.append(e_t)
        rel_error.append(e_t/T_norm)

        if writer is not None:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import Dataset, DataLoader
            from torch.utils.tensorboard import SummaryWriter
            writer.add_scalar('Relative Error', e_t/T_norm, epoch)

        if (epoch>0) and (e[epoch-1] - e[epoch] < tol*T_norm):
            success = True
            message = None
            #return a, b, d,rel_error,success,message
            print(epoch)
            return A, B, D, B_prime, D_prime, success, message
    success = False
    message = f'Reached # iterations: {max_iter}'
    return A, B, D, B_prime, D_prime, success, message