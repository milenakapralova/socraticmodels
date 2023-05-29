"""
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
"""

# Package loading
import os
import numpy as np
import pandas as pd
import sys

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass

# Local imports
from scripts.image_captioning import ImageCaptionerImproved

if __name__ == '__main__':
    image_captioner = ImageCaptionerImproved(n_images=50, set_type='train')
    image_captioner.random_parameter_search(n_iterations=100, n_captions=10)

