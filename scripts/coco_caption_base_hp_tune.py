"""
SocraticFlanT5 - Caption Generation (baseline) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the baseline Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is replaced by FLAN-T5-xl.
"""

# Package loading
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.image_captioning import ImageCaptionerBaseline


if __name__ == '__main__':
    image_captioner = ImageCaptionerBaseline(n_images=50, set_type='train')
    image_captioner.random_parameter_search(n_iterations=200, n_captions=10)