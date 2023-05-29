"""
SocraticGPT-3 - Caption Generation | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the original Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is used as a LLM.

Set-up
If you haven't done so already, please activate the corresponding environment by running in the terminal:
`conda env create -f environment.yml`. Then type `conda activate socratic`.
"""

# Package loading
import sys
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.coco_caption_base_hp_tune import ImageCaptionerBaseline



if __name__ == '__main__':
    image_captioner = ImageCaptionerGTP(n_images=50, set_type='test')
    caption_dir = '../data/outputs/captions/'
    run_params = {
        'n_captions': 10,
        'lm_temperature': 0.9,
        'caption_strategy': 'gpt'
    }
    image_captioner.main(**run_params)
