import time
import torch
import numpy as np
import os


def print_time_dec(func):
    def wrap(*args, **kwargs):
        start = time.time()
        print(f'{func.__name__} starting!')
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {np.round(end - start, 1)}s!')
        return result
    return wrap


def get_device():
    # Set the device to use
    if getattr(torch, 'has_mps', False):
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def prepare_dir(file_path):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    dir_path = '/'.join(file_path.split('/')[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass