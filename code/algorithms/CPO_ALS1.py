import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# # Fix tensorboard problem
# import tensorflow as tf
# import tensorboard as tb

def cpo_als1(T, T_norm, rank, a, b, d, writer, max_iter=50, tol=1e-4):
    # decompose with orthogonality constraint on b and d
    # simplification taken from Sorensen: CANONICAL POLYADIC DECOMPOSITION WITH A COLUMNWISE ORTHONORMAL FACTOR MATRIX

    # a = np.random.random((tensor.shape[0], rank))
    # b = np.random.random((tensor.shape[1], rank))
    # d = np.random.random((tensor.shape[2], rank))

    e = []
    rel_error = []

    for epoch in range(max_iter):
        # optimize b
        U, E, V_T = np.linalg.svd(khatri_rao([d, a]).T@f_unfold(T,1).T, full_matrices=False)
        b = (U@V_T).T
        assert np.allclose(b.T@b, np.eye(b.shape[1]))

        # optimize d
        U, E, V_T = np.linalg.svd(khatri_rao([b, a]).T@f_unfold(T,2).T, full_matrices=False)
        d = (U@V_T).T
        assert np.allclose(d.T@d, np.eye(d.shape[1]))

        # optimize a
        a = f_unfold(T,0)@khatri_rao([d, b])

        a_b_d = tl.cp_to_tensor((np.array([1]*rank), [a,b,d]))
        e_t = np.linalg.norm(T - a_b_d)
        e.append(e_t)
        rel_error.append(e_t/T_norm)

        if writer is not None:
            writer.add_scalar('Relative Error', e_t/T_norm, epoch)
        
        if (epoch>0) and (e[epoch] - e[epoch-1] < tol*T_norm):
            success = True
            message = None
            return a, b, d,rel_error,success,message
    success = False
    message = f'Reached # iterations: {max_iter}'
    return a, b, d,rel_error,success,message