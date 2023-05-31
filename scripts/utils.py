"""
The utils.py file contains the following utilities function:
- print_time_dec
- get_device
- prepare_dir
- get_uuid_for_imgs
Please refer to the function docstrings for more information.
"""

import time
import torch
import numpy as np
import os
import uuid
import json
from transformers import set_seed


def print_time_dec(func):
    """
    This function is used to decorate the functions/methods that we want to time.

    It will first start that the function is starting. Record the time. At the end of the execution, the
    runtime is printer out

    :param func:
    :return:
    """
    def wrap(*args, **kwargs):
        start = time.time()
        print(f'{func.__name__} starting!')
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {np.round(end - start, 1)}s!')
        return result
    return wrap


def get_device():
    """
    This function determines the device to use.

    If the user has a M1/M2 chip, it will use mps.
    If the use has a cuda gpu, it will use cuda.
    Otherwise, it will use the cpu.

    :return:
    """
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
    # Remove the file name to only keep the directory path.
    dir_path = '/'.join(file_path.split('/')[:-1])
    # Try to create the directory. Will have no effect if the directory already exists.
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def set_all_seeds(num):
    """
    This function sets the seeds and ensures reproducible results.
    """
    # Seed for numpy.
    np.random.seed(num)

    # Seed for the HuggingFace transformer library.
    set_seed(num)


def get_uuid_for_imgs(img_list):
    """
    This function receives a list of string and creates a deterministic UUID identifier.

    The purpose of this function is to get a unique identifier so that a set of images embeddings
    and their captions can be cached in order to speed up development time.

    :param img_list: The list of images.
    :return: A UUID string as a unique identifier for the image list.
    """
    # The sort enforces the list order
    img_list.sort()
    # Only the exact same list of images will lead to the same UUID.
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(img_list)))


def get_samples_sqa(sample_idxs_path='../data/scienceqa/sample_idxs.json'):
    """
    Saves a list of sample indices for each task in ScienceQA dataset.
    """
    sqa_samples = {
        'cot': [
            68, 90, 122, 142, 155, 167, 177, 191, 202, 227, 234, 236, 254, 325, 340, 352, 364, 369, 370, 372, 374, 376,
            426, 429, 430, 432, 433, 448, 449, 461, 472, 473, 474, 475, 501, 502, 506, 513, 523, 529, 530, 534, 544,
            553, 557, 559, 570, 574, 575, 601, 602, 606, 607, 609
        ],
        'vqa': [
            135, 140, 148, 155, 177, 202, 215, 223, 227, 234, 236, 237, 254, 257, 301, 307, 310, 311, 316, 319, 320,
            324, 325, 327, 331, 334, 342, 348, 352, 360, 363, 364, 368, 369, 372, 400, 406, 426, 427, 433, 436, 449,
            461, 470, 491, 494, 506, 507, 508, 509, 513, 523, 527, 530, 540, 545, 550, 557, 558, 568, 574, 583, 585,
            588, 603, 605
        ]
    }
    # save samples to file
    prepare_dir(sample_idxs_path)
    with open(sample_idxs_path, 'w') as f:
        json.dump(sqa_samples, f)
