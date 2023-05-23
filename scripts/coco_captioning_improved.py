'''
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the improved Socratic model pipeline:
an improved baseline model where the template prompt filled by CLIP is processed before passing to the LM.

'''

# Package loading
import os
import pandas as pd
import sys
sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass
import argparse
import json
# Local imports
import scripts.image_captioning as ic
from scripts.utils import get_device, set_all_seeds, get_file_name_extension, print_time_dec


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

    # dict to store image info
    # img_dict = dict.fromkeys(['name', 'img', 'feats', 'img_type', 'num_ppl', 'location', 'objs'])
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
        
        # filter unique objects
        if args.filter_mode == 'default':
            filtered_objs = ic.filter_objs(sorted_objs, obj_scores, clip_manager, args.obj_topk, args.sim_threshold)
        elif args.filter_mode == 'alt':
            filtered_objs = ic.filter_objs_alt(vocab_manager.object_list, sorted_objs, obj_emb, img_feats, clip_manager, args.obj_topk, args.sim_threshold)
        else:
            raise ValueError(f'Invalid filter mode: {args.filter_mode}. Must be one of: default, alt')
        print(f'filtered objects: {filtered_objs}')
        
        if args.verbose:
            print(f'img type: {img_type} | # ppl: {num_ppl} | locations: {locations}\n | objs: {filtered_objs}\n')
        
         # generate prompt
        if args.caption_mode == 'baseline':
            prompt = prompt_generator.create_baseline_lm_prompt(img_type, num_ppl, locations, filtered_objs) 
        elif args.caption_strategy == 'improved':
            prompt = prompt_generator.create_improved_lm_prompt(img_type, num_ppl, locations, filtered_objs)
        else:
            raise ValueError(f'Invalid caption mode: {args.caption_mode}. Must be one of: baseline, improved')
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
        
    avg_cos_sim = sum([res['cos_sim'] for res in results]) / args.num_imgs
    print(f'avg. cos sim over {args.num_imgs} imgs: {avg_cos_sim:.4f}')
    
    # save results & params
    print('done')
    res_dir = f'{args.output_dir}/{args.lm_version}/'
    os.makedirs(res_dir, exist_ok=True)
    output_file_suffix = get_file_name_extension(args.temperature, args.sim_threshold, args.obj_topk, args.places_topk, args.caption_mode) if args.output_file_suffix is None else args.output_file_suffix
    res_path = f'{res_dir}/improved_captions{output_file_suffix}.csv'
    args_path = f'{res_dir}/improved_params{output_file_suffix}.json'
    pd.DataFrame(results).to_csv(res_path, index=False)
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)

if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Image Captioning')
    # add args
    parser.add_argument('--output-dir', type=str, default='../outputs/captions', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default=None, help='suffix for output file')
    parser.add_argument('--num-imgs', type=int, default=50, help='# imgs to sample randomly from MS-COCO')
    parser.add_argument('--num-captions', type=int, default=10, help='# captions to generate per img')
    parser.add_argument('--rand-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument('--obj-topk', type=int, default=10, help='# top objects detected to keep per img (CLIP)')
    parser.add_argument('--places-topk', type=int, default=3, help='# top places detected to keep per img (CLIP)')
    parser.add_argument('--sim-threshold', type=float, default=0.7, help='cosine similarity threshold for filtering objects')
    parser.add_argument('--caption-mode', type=str, default='default', help='caption strategy to use')
    parser.add_argument('--filter-mode', type=str, default='baseline', help='method to use for filtering objects')
    parser.add_argument('--param-search', type=bool, default=False, help='whether to run parameter search')
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print intermediate results')

    # LM params
    parser.add_argument('--lm-version', type=str, default='google/flan-t5-xl', help='name of the LM model to use on HuggingFace')
    parser.add_argument('--use-api', type=bool, default=False, help='whether to use the HuggingFace API for inference')
    parser.add_argument('--temperature', type=float, default=1., help='temperature param for inference')
    parser.add_argument('--max-length', type=int, default=None, help='max length (tokens) of generated caption')
    parser.add_argument('--min-length', type=int, default=None, help='minimum length (tokens) of generated caption')
    parser.add_argument('--max-new-tokens', type=int, default=30, help='max amount of new tokens to be generated, not including input tokens')
    parser.add_argument('--min-new-tokens', type=int, default=None, help='minimum amount of new tokens to be generated, not including input tokens')
    parser.add_argument('--num-beams', type=int, default=16, help='# beams for beam search')
    parser.add_argument('--do-sample', type=bool, default=True, help='whether to use sampling during generation')
    parser.add_argument('--num-return-sequences', type=int, default=1, help='# of sequences to generate')
    parser.add_argument('--early-stopping', type=bool, default=True, help='whether to enable early stopping')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3, help='size of n-grams to avoid repeating')
    parser.add_argument('--length-penalty', type=float, default=2.0, help='length penalty applied during generation')

    # call main with args
    args = parser.parse_args()
    print(f'args: {args}')

    # Run with the base parameters
    main(args)

    # parameter search
    if args.param_search:
        # temperature
        for t in (0.85, 0.95):
            temp_args = args
            temp_args.temperature = t
            if args.output_file_suffix is not None:
                temp_args.output_file_suffix = args.output_file_suffix + f'_temp_{t}'
            main(temp_args)

        # cosine similarity threshold search
        for c in (0.6, 0.8):
            cos_args = args
            cos_args.sim_threshold = c
            if args.output_file_suffix is not None:
                cos_args.output_file_suffix = args.output_file_suffix + f'_cos_{c}'
            main(cos_args)

        # object topk search
        for obj_k in (4, 6, 7):
            obj_args = args
            obj_args.objs_topk = obj_k
            if args.output_file_suffix is not None:
                obj_args.output_file_suffix = args.output_file_suffix + f'_obj_{obj_k}'
            main(obj_args)

        # places topk search
        for places_k in (1, 3):
            place_args = args
            place_args.places_topk = places_k
            if args.output_file_suffix is not None:
                place_args.output_file_suffix = args.output_file_suffix + f'_places_{places_k}'
            main(place_args)

