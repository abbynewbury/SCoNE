import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao

def cpo_als1(T, T_norm, rank, A, B, D, writer, max_iter=100, tol=1e-4):
    # decompose with orthogonality constraint on B and D
    # simplification taken from Sorensen: CANONICAL POLYADIC DECOMPOSITION WITH A COLUMNWISE ORTHONORMAL FACTOR MATRIX


    e = []
    rel_error = []

    for epoch in range(max_iter):
        # optimize b
        U, E, V_T = np.linalg.svd(khatri_rao([D, A]).T@f_unfold(T,1).T, full_matrices=False)
        B = (U@V_T).T
        assert np.allclose(B.T@B, np.eye(B.shape[1]))

        # optimize d
        U, E, V_T = np.linalg.svd(khatri_rao([B, A]).T@f_unfold(T,2).T, full_matrices=False)
        D = (U@V_T).T
        assert np.allclose(D.T@D, np.eye(D.shape[1]))

        # optimize a
        A = f_unfold(T,0)@khatri_rao([D, B])

        A_B_D = tl.cp_to_tensor((np.array([1]*rank), [A,B,D]))
        e_t = np.linalg.norm(T - A_B_D)
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
            return A, B, D,rel_error,success,message
    success = False
    message = f'Reached # iterations: {max_iter}'
    return A, B, D,rel_error,success,message