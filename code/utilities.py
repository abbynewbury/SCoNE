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
def vec2mats(v, shapes): 
    """ Converts a vector into matrices based on given shapes.
    shapes is list of tuples defining shapes of each matrix to be formed
    """
    place_sum = 0
    mats = []
    for dim in shapes:
        size = dim[0]*dim[1]
        mats.append(v[place_sum:place_sum+size].reshape(dim[0], dim[1], order='F'))
        place_sum += size
    return mats

#As mats2vec per Kolda textbook
def mats2vec(mats):
    """ Converts matrices back to vector form."""
    return np.concatenate([mat.flatten(order='F') for mat in mats])

# How our tensor is constructed: MZ_X: SNP data (w/ or w/out covs residualized), MZ_C: clinical data (w/ or w/out covs residualized)
def tensor_func(i, j, k, MZ_X, MZ_C):
    return MZ_X[i,j] * MZ_C[i,k]

# Algorithm comparison functions
def profile_function(func, *args, mem_target='function', **kwargs):
    gc.collect()
    start_time = time.process_time()
    start_user_time = os.times().user
    if mem_target == 'function':
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elif mem_target == 'subprocess':
        start_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    result = func(*args, **kwargs)  # Run the function
    gc.collect()

    if mem_target == 'function':
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:  # 'subprocess'
        end_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    max_mem = (end_mem - start_mem) / 1024  # KB to MB
    cpu_time = time.process_time() - start_time
    user_time = os.times().user - start_user_time

    if isinstance(result, tuple):
        return (*result, max_mem, cpu_time, user_time)
    else:
        return result, max_mem, cpu_time, user_time
    

# Other helper functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))