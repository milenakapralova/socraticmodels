'''
SocraticFlanT5 - Caption Generation (baseline) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the baseline Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is replaced by FLAN-T5-xl.

'''

# Package loading
import sys

# Local imports
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except:
    pass
from scripts.coco_captioning_baseline import ImageCaptionerBaseline


if __name__ == '__main__':
    image_captioner = ImageCaptionerBaseline(n_images=50, set_type='test')
    caption_dir = '../data/outputs/captions/'
    file_list = [
        f for f in os.listdir(caption_dir)
        if f.startswith('baseline') and f.endswith('csv') and 'train' in f
    ]
    for f in file_list:
        str_split = f.split('_')
        run_params = {
            'n_captions': 10,
            'lm_temperature': float(f'0.{str_split[3][1:]}'),
            'lm_do_sample': True,
            'n_objects': int(str_split[5]),
            'n_places': int(str_split[7]),
            'caption_strategy': str_split[9]
        }
        image_captioner.main(**run_params)
