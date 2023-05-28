'''
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the improved Socratic model pipeline:
an improved baseline model where the template prompt filled by CLIP is processed before passing to FLAN-T5-xl

'''

# Package loading
import os
import sys
sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass

# Local imports
from scripts.coco_captioning_improved import ImageCaptionerImproved


if __name__ == '__main__':
    image_captioner = ImageCaptionerImproved(n_images=50, set_type='test')
    caption_dir = '../data/outputs/captions/'
    file_list = [
        f for f in os.listdir(caption_dir)
        if f.startswith('improved') and f.endswith('csv') and 'train' in f
    ]
    for f in file_list:
        str_split = f.split('_')
        run_params = {
            'n_captions': 10,
            'lm_temperature': float(f'0.{str_split[3][1:]}'),
            'lm_max_length': 40,
            'lm_do_sample': True,
            'cos_sim_thres': float(f'0.{str_split[5][1:]}'),
            'n_objects': int(str_split[7]),
            'n_places': int(str_split[9]),
            'caption_strategy': str_split[11]
        }
        image_captioner.main(**run_params)
