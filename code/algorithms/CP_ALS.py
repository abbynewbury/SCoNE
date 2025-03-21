import numpy as np
from utilities import f_unfold
import tensorly as tl
from tensortools.operations import khatri_rao
import math

def cp_als(T, T_norm, rank, a, b, d, max_iter=50, tol=1e-4):
    # decompose with orthogonality constraint on b and d - similar to Algorithm 11.1 in Kolda textbook

    # a = np.random.random((tensor.shape[0], rank))
    # b = np.random.random((tensor.shape[1], rank))
    # d = np.random.random((tensor.shape[2], rank))

    S2 = b.T@b
    S3 = d.T@d

    e = []
    rel_error = []

    for epoch in range(max_iter):
        # optimize a
        # solving AV1 = U1
        V1 = np.multiply(S2, S3) # no reg param here
        U1 = f_unfold(T,0)@khatri_rao([d,b])
        a = np.linalg.solve(V1.T,U1.T).T
        S1 = a.T@a

        # optimize b
        # solving BV2=U2
        V2 = np.multiply(S3, S1)
        U2 = f_unfold(T,1)@khatri_rao([d,a])
        b = np.linalg.solve(V2.T,U2.T).T
        S2 = b.T@b

        # optimize d
        # solving DV3=U3
        V3 = np.multiply(S2,S1)
        U3 = f_unfold(T,2)@khatri_rao([b,a])
        d = np.linalg.solve(V3.T,U3.T).T
        S3 = d.T@d

        alpha = np.tensordot(T, tl.cp_to_tensor((np.array([1]*rank), [a,b,d])), axes=T.ndim) # inner prod. between T and [[A,B,D]] - think formula in Alg 11.1 is off
        beta = np.ones(rank).T@np.multiply(S3,V3)@np.ones(rank)# squared norm of [[A,B,D]]
        e_t = math.sqrt(T_norm**2 - 2*alpha + beta) # ||T-[[A,B,D]]||
        #a_b_d = tl.cp_to_tensor((np.array([1]*rank), [a,b,d]))
        e.append(e_t)
        rel_error.append(e_t/T_norm)

        if (epoch>0) and (e[epoch] - e[epoch-1] < tol*T_norm):
            success = True
            message = None
            return a, b, d,rel_error,success,message
    success = False
    message = f'Reached # iterations: {max_iter}'
    return a, b, d,rel_error,success,message