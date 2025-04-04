import numpy as np
import resource
import time
import os
import gc

# Tensor helper functions
# define tensor unfolding per Kolda textbook
def f_unfold(tensor, mode=0):
    """Unfolds a tensors following the Kolda and Bader definition

        Moves the `mode` axis to the beginning and reshapes in Fortran order
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), 
                      (tensor.shape[mode], -1), order='F')

def f_refold(tensor, original_shape, mode=0):
    # mode is mode that it was unfolded into
    return np.moveaxis(np.reshape(tensor, np.roll(original_shape, -mode), order='F'), 0, mode)


# As vec2mats per Kolda textbook
def vec2mats(v, rank, T_0, T_1, T_2): 
    """ Converts a vector into matrices A, B, D based on given shapes.
    T_0: shape of first mode of T
    T_1: shape of second mode of T
    T_2: shape of third mode of T
    """
    A = v[:T_0*rank].reshape(T_0, rank, order='F')
    B = v[T_0*rank:T_0*rank+T_1*rank].reshape(T_1, rank, order='F')
    D = v[T_0*rank+T_1*rank:].reshape(T_2, rank, order='F')
    return A, B, D

#As mats2vec per Kolda textbook
def mats2vec(G1, G2, G3):
    """ Converts matrices back to vector form."""
    return np.concatenate([G1.flatten(order='F'), G2.flatten(order='F'), G3.flatten(order='F')])

# How our tensor is constructed: MZ_X: SNP data (w/ or w/out covs residualized), MZ_C: clinical data (w/ or w/out covs residualized)
def tensor_func(i, j, k, MZ_X, MZ_C):
    return MZ_X[i,j] * MZ_C[i,k]

# Algorithm comparison functions
def profile_function(func, *args, **kwargs):
    gc.collect()
    start_time = time.process_time()
    start_user_time = os.times().user
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = func(*args, **kwargs)  # Run the function
    gc.collect()
    max_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem) / 1024  # Convert KB to MB
    cpu_time = time.process_time() - start_time
    user_time = os.times().user - start_user_time

    return *result, max_mem, cpu_time, user_time