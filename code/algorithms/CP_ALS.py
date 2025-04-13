import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import math


def cp_als(T, T_norm, rank, A, B, D, writer, max_iter=100, tol=1e-4):
    # similar to Algorithm 11.1 in Kolda textbook
    # writer is tensorboard writer


    S2 = B.T@B
    S3 = D.T@D

    e = []
    rel_error = []

    for epoch in range(max_iter):
        # optimize a
        # solving AV1 = U1
        V1 = np.multiply(S2, S3) # no reg param here
        U1 = f_unfold(T,0)@khatri_rao([D,B])
        A = np.linalg.solve(V1.T,U1.T).T
        S1 = A.T@A

        # optimize b
        # solving BV2=U2
        V2 = np.multiply(S3, S1)
        U2 = f_unfold(T,1)@khatri_rao([D,A])
        B = np.linalg.solve(V2.T,U2.T).T
        S2 = B.T@B

        # optimize d
        # solving DV3=U3
        V3 = np.multiply(S2,S1)
        U3 = f_unfold(T,2)@khatri_rao([B,A])
        D = np.linalg.solve(V3.T,U3.T).T
        S3 = D.T@D

        alpha = np.tensordot(T, tl.cp_to_tensor((np.array([1]*rank), [A,B,D])), axes=T.ndim) # inner prod. between T and [[A,B,D]] - think formula in Alg 11.1 is off
        beta = np.ones(rank).T@np.multiply(S3,V3)@np.ones(rank)# squared norm of [[A,B,D]]
        e_t = math.sqrt(T_norm**2 - 2*alpha + beta) # ||T-[[A,B,D]]||
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
            return A, B, D, success, message
    success = False
    message = f'Reached # iterations: {max_iter}'
    return A, B, D, success, message