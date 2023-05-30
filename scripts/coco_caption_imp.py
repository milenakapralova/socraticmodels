"""
This file is used to generate captions for the Coco dataset using the improved image captioner.

For more info on the ImageCaptionerImproved class, please check out the docstrings in the image_captioning.py file.
"""

# Package loading
import os
import sys
sys.path.append('..')
import argparse
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.coco_caption_imp_hp_tune import ImageCaptionerImproved


def parse_arguments():
    """
    Parses the arguments for the improved COCO captioning.

    :return:
    """
    # init argparser
    parser = argparse.ArgumentParser(description='Improved Image Captioning')

    # add args
    parser.add_argument('--n-images', type=int, default=50, help='# images to include in the dataset')
    parser.add_argument('--set-type',  type=str, default='train', help='train/valid/test set')
    parser.add_argument('--n-captions', type=int, default=10, help='# captions the LM should generate')
    parser.add_argument('--lm-temperature', type=float, default=0.9, help='temperature param for the lm')
    parser.add_argument('--lm-max-length', type=int, default=40, help='max output length the LM should generate')
    parser.add_argument('--lm-do-sample', type=bool, default=True, help='whether to use sampling during generation')
    parser.add_argument('--cos-sim-thres', type=float, default=0.8, help='temperature param for the lm')
    parser.add_argument('--n-objects', type=int, default=10, help='# objects to include in the LM prompt')
    parser.add_argument('--n-places', type=int, default=3, help='# places to include in the LM prompt')
    parser.add_argument('--caption-strategy',  type=str, default='original', help='caption strategy to use')

    # parse args
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # Parse the arguments.
    args = parse_arguments()

    # Instantiate the improved image captioner class.
    image_captioner = ImageCaptionerImproved(n_images=args.n_images, set_type=args.set_type)

    # Run the main method to produce the captions
    image_captioner.main(
        n_captions=args.n_captions, lm_temperature=args.lm_temperature, lm_max_length=args.lm_max_length,
        lm_do_sample=args.lm_do_sample, cos_sim_thres=args.cos_sim_thres, n_objects=args.n_objects,
        n_places=args.n_places, caption_strategy=args.caption_strategy
    )
