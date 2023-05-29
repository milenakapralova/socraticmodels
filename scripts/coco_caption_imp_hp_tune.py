"""
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
"""

# Package loading
import os
import sys
import argparse

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
sys.path.append('..')
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.image_captioning import ImageCaptionerImproved


def parse_arguments():
    # init argparser
    parser = argparse.ArgumentParser(description='Baseline Image Captioning Hyperparameter tuning')

    # Additional variables
    parser.add_argument('--n-images', type=int, default=50, help='# images to include in the dataset')
    parser.add_argument('--set-type',  type=str, default='train', help='train/valid/test set')
    parser.add_argument('--n-iterations', type=int, default=100, help='# of run iterations')
    parser.add_argument('--n-captions', type=int, default=10, help='# captions the LM should generate')
    parser.add_argument('--lm-max-length', type=int, default=40, help='max output length the LM should generate')
    parser.add_argument('--lm-do-sample', type=bool, default=True, help='whether to use sampling during generation')
    parser.add_argument('--lm-temp-min', type=float, default=0.5, help='minimum temperature param for the lm')
    parser.add_argument('--lm-temp-max', type=float, default=1, help='maximum temperature param for the lm')
    parser.add_argument(
        '--cos-sim-thres-min', type=float, default=0.6, help='min cosine sim threshold for the improved Socratic'
    )
    parser.add_argument(
        '--cos-sim-thres-max', type=float, default=1, help='max cosine sim threshold for the improved Socratic'
    )
    parser.add_argument('--n-objects-min', type=int, default=5, help='minimum number of objects in the LM prompt')
    parser.add_argument('--n-objects-max', type=int, default=15, help='maximum number of objects in the LM prompt')
    parser.add_argument('--n-places-min', type=int, default=1, help='minimum number of places in the LM prompt')
    parser.add_argument('--n-places-max', type=int, default=6, help='maximum number of places in the LM prompt')
    parser.add_argument('--caption-strategies', nargs="+", default=None)

    # parse args
    args = parser.parse_args()

    return args



if __name__ == '__main__':

    # Parse the arguments.
    args = parse_arguments()

    image_captioner = ImageCaptionerImproved(n_images=args.n_images, set_type=args.set_type)

    # Run the hyperparameter search
    image_captioner.random_parameter_search(
        n_iterations=args.n_iterations, n_captions=args.n_captions, lm_max_length=args.lm_max_length,
        lm_do_sample=args.lm_do_sample, lm_temp_min=args.lm_temp_min, lm_temp_max=args.lm_temp_max,
        cos_sim_thres_min=args.cos_sim_thres_min, cos_sim_thres_max=args.cos_sim_thres_max,
        n_objects_min=args.n_objects_min, n_objects_max=args.n_objects_max, n_places_min=args.n_places_min,
        n_places_max=args.n_places_max, caption_strategies=args.caption_strategies
    )
