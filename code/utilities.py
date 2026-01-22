import numpy as np
import resource
import time
import os
import gc
from pathlib import Path
import shutil


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
    
