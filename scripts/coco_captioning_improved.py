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
    lm_manager = ic.LmManager(version=args.lm_model, use_api=args.use_api, device=device)
    cache_manager = ic.CacheManager()
    
    # instantiate prompt generator
    prompt_generator = ic.LmPromptGenerator()

    # compute place & object features
    place_emb = cache_manager.get_place_emb(clip_manager, vocab_manager)
    object_emb = cache_manager.get_object_emb(clip_manager, vocab_manager)

    # randomly select images from the COCO dataset
    img_files = coco_manager.get_random_image_paths(num_images=args.num_imgs)

    # dict to store image info
    img_dict = dict.fromkeys(['name', 'img', 'feats', 'img_type', 'num_ppl', 'location', 'objs'])
    # list of dicts to store results
    results = []    
    
    # set LM params
    lm_params = {"min_new_tokens": args.min_new_tokens, "max_new_tokens": args.max_new_tokens, "length_penalty": args.length_penalty, "num_beams": args.num_beams, "no_repeat_ngram_size": args.no_repeat_ngram_size, "temperature": args.temperature,  "early_stopping": args.early_stopping, "do_sample": args.do_sample, "num_return_sequences": args.num_return_sequences}
    
    '''2. Generate captions for each image'''
    
    for img_file in img_files:
        # load  image
        img_dict['name'] = img_file
        img = image_manager.load_image(coco_manager.image_dir + img_file)
        img_dict['img'] = img
        # generate the CLIP image embedding
        img_feats = clip_manager.get_img_emb(img_dict['img']).flatten()
        img_dict['feats'] = img_feats

        # get image info (type, # ppl, location, objects) using CLIP w/ zero-shot classification
        img_type, num_ppl, locations, sorted_objs, topk_objs, obj_scores = clip_manager.get_img_info(img, place_emb, object_emb, vocab_manager, args.obj_topk)
        img_dict['img_type'] = img_type
        img_dict['num_ppl'] = num_ppl
        img_dict['locations'] = locations
        
        # filter unique objects
        filtered_objs = ic.filter_objs(sorted_objs, obj_scores, clip_manager, obj_topk=10, sim_threshold=args.sim_threshold)
        # filtered_objs = ic.filter_objs_alt(vocab_manager.object_list, sorted_obj_texts, obj_feats, img_feats, clip_manager, obj_top=10)
        print(f'filtered objects: {filtered_objs}')
        img_dict['objs'] = filtered_objs
        
         # generate prompt
        prompt = prompt_generator.create_baseline_lm_prompt(img_type, num_ppl, locations, topk_objs)
        
        # generate captions by propmting LM (zero-shot)
        caption_texts = lm_manager.generate_response(args.num_captions * [prompt], lm_params)
        
        # rank captions by CLIP
        sorted_captions = clip_manager.rank_gen_outputs(img_feats, caption_texts)
        best_caption, best_score = next(iter(sorted_captions.items()))
        
        # store best caption & score
        results.append({
            'img_name': img_file,
            'best_caption': best_caption,
            'cos_sim': best_score,
        })
    
    # save results & params
    res_dir = f'{args.output_dir}/{args.lm_model}/'
    os.makedirs(res_dir, exist_ok=True)
    res_path = f'{res_dir}/res_improved_{args.output_file_suffix}.csv'
    # file_name_extension = get_file_name_extension(
    #     args.temperature, args.sim_threshold, args.num_objects, args.num_places, args.caption_strategy
    # )
    # file_path = f'../data/outputs/captions/improved_caption{file_name_extension}.csv'
    pd.DataFrame(results).to_csv(res_path, index=False)
    args_path = f'{res_dir}/params_improved_{args.output_file_suffix}.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)

if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Image Captioning')
    # add args
    parser.add_argument('--output-dir', type=str, default='../outputs/captions', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default='1', help='suffix for output file')
    parser.add_argument('--num-imgs', type=int, default=50, help='# imgs to sample randomly from MS-COCO')
    parser.add_argument('--num-captions', type=int, default=10, help='# captions to generate per img')
    parser.add_argument('--rand-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument('--obj-topk', type=int, default=10, help='# top objects detected to keep per img (CLIP)')
    parser.add_argument('--places-topk', type=int, default=3, help='# top places detected to keep per img (CLIP)')
    parser.add_argument('--sim-threshold', type=float, default=0.7, help='cosine similarity threshold for filtering objects')
    parser.add_argument('--caption-strategy', type=str, default='baseline', help='caption strategy to use')
    parser.add_argument('--param-search', type=bool, default=False, help='whether to run parameter search')

    # LM params
    parser.add_argument('--lm-model', type=str, default='google/flan-t5-xl', help='name of the LM model to use on HuggingFace')
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
    print(f'args: {args}')

    # Run with the base parameters
    main(args)

    # parameter search
    if args.param_search:
        # Temperature search
        for t in (0.85, 0.95):
            temp_args = args
            temp_args.temperature = t
            temp_args.output_file_suffix = f'{args.output_file_suffix}_temp_{t}'
            main(temp_args)

        # Cosine similarity threshold search
        for c in (0.6, 0.8):
            temp_args = args
            temp_args.sim_threshold = c
            temp_args.output_file_suffix = f'{args.output_file_suffix}_sim_{c}'
            main(temp_args)

        # Cosine similarity threshold search
        for n in (4, 6, 7):
            temp_args = args
            temp_args.objs_topk = n
            temp_args.output_file_suffix = f'{args.output_file_suffix}_objs_{n}'
            main(temp_args)

        # Cosine similarity threshold search
        for n in (1, 3):
            temp_args = args
            temp_args.places_topk = n
            temp_args.output_file_suffix = f'{args.output_file_suffix}_places_{n}'
            main(temp_args)

