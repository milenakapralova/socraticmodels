"""
SocraticGPT-3 - Caption Generation | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the original Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is used as a LLM.

Set-up
If you haven't done so already, please activate the corresponding environment by running in the terminal:
`conda env create -f environment.yml`. Then type `conda activate socratic`.
"""
import argparse
# Package loading
import sys
sys.path.append('..')
import os

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.image_captioning import ImageCaptionerGPT


def parse_arguments():
    # init argparser
    parser = argparse.ArgumentParser(description='Baseline Image Captioning')

    # add args
    parser.add_argument('--n-images', type=int, default=50, help='# images to include in the dataset')
    parser.add_argument('--set-type',  type=str, default='train', help='train/valid/test set')
    parser.add_argument('--n-captions', type=int, default=10, help='# captions the LM should generate')
    parser.add_argument('--lm-temperature', type=float, default=0.9, help='temperature param for the lm')
    parser.add_argument('--lm-max-length', type=int, default=40, help='max output length the LM should generate')
    parser.add_argument('--lm-do-sample', type=bool, default=True, help='whether to use sampling during generation')
    parser.add_argument('--n-objects', type=int, default=10, help='# objects to include in the LM prompt')
    parser.add_argument('--n-places', type=int, default=3, help='# places to include in the LM prompt')
    parser.add_argument('--caption-strategy',  type=str, default='original', help='caption strategy to use')

    # parse args
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # Parse the arguments.
    args = parse_arguments()

    # Instantiate the gpt image captioner class.
    image_captioner = ImageCaptionerGPT(n_images=args.n_images, set_type=args.set_type)

    # Run the main method to produce the captions
    image_captioner.main(
        n_captions=args.n_captions, lm_temperature=args.lm_temperature, lm_max_length=args.lm_max_length,
        lm_do_sample=args.lm_do_sample, n_objects=args.n_objects, n_places=args.n_places,
        caption_strategy=args.caption_strategy
    )
