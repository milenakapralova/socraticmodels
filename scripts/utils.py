import time
import torch
import numpy as np
import os
import uuid
from transformers import set_seed

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
    This function is used to create the directories needed to outputs a path. If the directories already exist, the
    function continues.
    """
    dir_path = '/'.join(file_path.split('/')[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def set_all_seeds(num):
    """
    This function sets the seeds and ensures reproducible results.
    """
    # Seed for selecting images from the validation split
    np.random.seed(num)

    # Seed for FLAN-T5
    set_seed(num)


def get_uuid_for_imgs(img_list):
    """
    This function receives a list of string and creates a deterministic UUID identifier.

    :param img_list: The list of images.
    :return: A UUID string as a unique identifier for the image list.
    """
    img_list.sort()
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(img_list)))


def get_file_name_extension_baseline(lm_temperature, num_objects, num_places):
    extension = ''
    extension += f'_temp_{lm_temperature}'.replace('.', '')
    extension += f'_nobj_{num_objects}'
    extension += f'_npl_{num_places}'
    return extension


def get_file_name_extension_improved(lm_temperature, cos_sim_thres, num_objects, num_places, caption_strategy, set_type):
    extension = ''
    # The language model temperature
    extension += f'_temp_{lm_temperature}'.replace('.', '')
    # The cosine thresold.
    extension += f'_costhres_{cos_sim_thres}'.replace('.', '')
    # Number of objects
    extension += f'_nobj_{num_objects}'
    # Number of places
    extension += f'_npl_{num_places}'
    # Caption strategy
    extension += f'_strat_{caption_strategy}'
    # Train/test set
    extension += f'_{set_type}'
    return extension

