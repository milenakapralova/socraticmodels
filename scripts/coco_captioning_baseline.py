'''
SocraticFlanT5 - Caption Generation (baseline) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the baseline Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is replaced by an open-source LM.

Set-up
If you haven't done so already, please activate the corresponding environment by running in the terminal:
`conda env create -f environment.yml`. Then type `conda activate socratic`.
'''

# Package loading
import pandas as pd
import sys
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except:
    pass
import argparse
import json

# Local imports
import scripts.image_captioning as ic
from scripts.utils import get_device, set_all_seeds, print_time_dec


@print_time_dec
def main(args):

    '''1. Set up'''

    # seed random seeds for reproducibility
    set_all_seeds(args.rand_seed)

    # download the MS COCO images and annotations
    coco_manager = ic.CocoManager()

    # set device
    device = get_device()

    # instantiate managers
    clip_manager = ic.ClipManager(device)
    image_manager = ic.ImageManager()
    vocab_manager = ic.VocabManager()
    lm_manager = ic.LmManager(version=args.lm_version, use_api=args.use_api, device=device)
    cache_manager = ic.CacheManager()
    
    # instantiate prompt generator
    prompt_generator = ic.LmPromptGenerator()

    # compute place & object features
    place_emb = cache_manager.get_place_emb(clip_manager, vocab_manager)
    obj_emb = cache_manager.get_object_emb(clip_manager, vocab_manager)

    # randomly select images from the COCO dataset
    img_fnames = coco_manager.get_random_image_paths(num_images=args.num_imgs)

    # list of dicts to store results
    results = []    
    
    # set LM params
    lm_params = {"min_new_tokens": args.min_new_tokens, "max_new_tokens": args.max_new_tokens, "length_penalty": args.length_penalty, "num_beams": args.num_beams, "no_repeat_ngram_size": args.no_repeat_ngram_size, "temperature": args.temperature,  "early_stopping": args.early_stopping, "do_sample": args.do_sample, "num_return_sequences": args.num_return_sequences}
    
    '''2. Generate captions for each image'''
    
    for img_idx, img_fname in enumerate(img_fnames):
        print(f'generating captions for img {img_idx + 1}/{len(img_fnames)}...')
        # load  image
        img = image_manager.load_image(coco_manager.image_dir + img_fname)
        # generate the CLIP image embedding
        img_feats = clip_manager.get_img_emb(img).flatten()

        # get image info (type, # ppl, location, objects) using CLIP w/ zero-shot classification
        img_type, num_ppl, locations, sorted_objs, topk_objs, obj_scores = clip_manager.get_img_info(img, place_emb, obj_emb, vocab_manager, args.obj_topk)
        if args.verbose:
            print(f'img type: {img_type} | # ppl: {num_ppl} | locations: {locations}\n | objs: {topk_objs}\n')
        # generate prompt
        prompt = prompt_generator.create_baseline_lm_prompt(img_type, num_ppl, locations, topk_objs)
        if args.verbose:
            print(f'prompt: {prompt}\n')
        # generate captions by propmting LM (zero-shot)
        caption_texts = lm_manager.generate_response(args.num_captions * [prompt], lm_params)
        
        # rank captions by CLIP
        sorted_captions = clip_manager.rank_gen_outputs(img_feats, caption_texts)
        best_caption, best_score = next(iter(sorted_captions.items()))
        if args.verbose:
            print(f'best caption: {best_caption} | score: {best_score}\n')
        
        # store best caption & score
        results.append({
            'image_name': img_fname,
            'best_caption': best_caption,
            'cos_sim': best_score,
        })
        
        print('=' * 50)
       
    print('done')
    # save results & params
    res_dir = f'{args.output_dir}/{args.lm_version}/'
    os.makedirs(res_dir, exist_ok=True)
    res_path = f'{res_dir}/baseline_captions{args.output_file_suffix}.csv'
    pd.DataFrame(results).to_csv(res_path, index=False)
    args_path = f'{res_dir}/baseline_params{args.output_file_suffix}.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)


if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Image Captioning')
    # add args
    parser.add_argument('--output-dir', type=str, default='../outputs/captions', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--num-imgs', type=int, default=50, help='# imgs to sample randomly from MS-COCO')
    parser.add_argument('--num-captions', type=int, default=10, help='# captions to generate per img')
    parser.add_argument('--rand-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument('--obj-topk', type=int, default=10, help='# top objects detected to keep per img (CLIP)')
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print intermediate results')

    # LM params
    parser.add_argument('--lm-version', type=str, default='google/flan-t5-xl', help='name of the LM model to use on HuggingFace')
    parser.add_argument('--use-api', type=bool, default=False, help='whether to use the HuggingFace API for inference')
    parser.add_argument('--temperature', type=float, default=0.9, help='temperature param for inference')
    parser.add_argument('--max-length', type=int, default=None, help='max length (tokens) of generated caption')
    parser.add_argument('--min-length', type=int, default=None, help='minimum length (tokens) of generated caption')
    parser.add_argument('--max-new-tokens', type=int, default=20, help='max amount of new tokens to be generated, not including input tokens')
    parser.add_argument('--min-new-tokens', type=int, default=None, help='minimum amount of new tokens to be generated, not including input tokens')
    parser.add_argument('--num-beams', type=int, default=8, help='# beams for beam search')
    parser.add_argument('--do-sample', type=bool, default=True, help='whether to use sampling during generation')
    parser.add_argument('--num-return-sequences', type=int, default=1, help='# of sequences to generate')
    parser.add_argument('--early-stopping', type=bool, default=True, help='whether to enable early stopping')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3, help='size of n-grams to avoid repeating')
    parser.add_argument('--length-penalty', type=float, default=2.0, help='length penalty applied during generation')

    # call main with args
    args = parser.parse_args()
    print(args)
    main(args)