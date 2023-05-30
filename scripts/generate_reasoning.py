# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import torch
import argparse
import time
from datasets import load_dataset
import image_captioning as ic
from utils import print_time_dec, get_samples_sqa
import openai

@print_time_dec
def main(args):
    '''1. Set up'''
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # instantiate managers
    clip_manager = ic.ClipManager(device)
    vocab_manager = ic.VocabManager()
    cache_manager = ic.CacheManager()
    if args.lm_model != 'gpt':
        lm_manager = ic.LmManager(version=args.lm_model, use_api=True, device=device)
    # instantiate prompt generator
    prompt_generator = ic.LmPromptGenerator()
    
    # compute place & object features
    place_feats = cache_manager.get_place_emb(clip_manager, vocab_manager)
    obj_feats = cache_manager.get_object_emb(clip_manager, vocab_manager)
    
    # load scienceQA dataset
    scienceQA_dataset = load_dataset('derek-thomas/ScienceQA', split='validation')
    # filter out samples with no image
    scienceQA_dataset = [sample for sample in scienceQA_dataset if sample['image'] is not None]
    # load sample indices file
    if not os.path.exists(args.data_path):
        get_samples_sqa(args.data_path)
    with open(args.data_path, 'r') as f:
        sample_idxs_file = json.load(f)
    sample_idxs = sample_idxs_file['cot'] if args.task in ['cot_zs', 'cot_fs'] else sample_idxs_file['vqa']
    np.random.seed(args.rand_seed)
    sample_idxs = np.random.choice(sample_idxs, args.num_samples, replace=False)
    samples = [scienceQA_dataset[int(idx)] for idx in sample_idxs]

    '''2. Generate outputs for each sample'''
    outputs = []
    print(f'generating {args.num_samples} samples for {args.task} task...')
    for i, sample in tqdm(enumerate(samples)):
        # generate prompt
        # zero-shot CoT
        if args.task == 'cot_zs':
            prompt = prompt_generator.create_cot_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        # few-shot CoT
        elif args.task == 'cot_fs':
            eg_sample = scienceQA_dataset[args.eg_idx]
            prompt = prompt_generator.create_cot_prompt(eg_sample, clip_manager, vocab_manager, place_feats, obj_feats) + f'{sample["solution"]}. So the answer is {sample["choices"][sample["answer"]]}\n' + prompt_generator.create_cot_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        # VQA
        elif args.task == 'vqa_zs':
            prompt = prompt_generator.create_vqa_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        elif args.task == 'vqa_fs':
            eg_sample = scienceQA_dataset[args.eg_idx]
            prompt = prompt_generator.create_vqa_prompt(eg_sample, clip_manager, vocab_manager, place_feats, obj_feats) + f'{sample["answer"]}\n' + prompt_generator.create_vqa_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        else:
            raise ValueError(f'Invalid task: {args.task}. Please choose from: cot_zs, cot_fs, vqa_zs, vqa_fs')

        # generate output
        if args.lm_model == 'gpt':
            lm_params = {'max_tokens': args.max_tokens, 'temperature': args.temperature}
            try:
                output = ic.get_response_gpt(prompt, **lm_params)
            except openai.error.RateLimitError:
                # sleep if API rate limit exceeded
                print('API rate limit exceeded,sleeping for 120s...')
                time.sleep(120)
                output = ic.get_response_gpt(prompt, **lm_params)
        else:
            lm_params = {'max_new_tokens': args.max_tokens, 'temperature': args.temperature, 'do_sample': False, 'length_penalty': 2.} 
            output = lm_manager.generate_response(prompt, lm_params)
        
        gt = sample['solution'] if args.task in ['cot_zs', 'cot_fs'] else sample['answer']
        outputs.append({
                'gt': gt,
                'gen': output
            })
        
        if args.verbose:
            print(f'{i+1}\nprompt: {prompt}\ngt: {gt}, gen: {output}\n')
        
    print('done.')
    # save results & params
    res_dir = f'{args.output_dir}/{args.lm_model}'
    os.makedirs(res_dir, exist_ok=True)
    res_path = f'{res_dir}/responses_{args.task}{args.output_file_suffix}.csv'
    pd.DataFrame(outputs).to_csv(res_path, index=False)
    print(f'results saved to: {res_path}')
    
if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Reasoning')
    # add args
    parser.add_argument('--task', type=str, default='cot_zs', help='task to run')
    parser.add_argument('--lm-model', type=str, default='gpt', help='language model to use')
    parser.add_argument('--data-path', type=str, default='../data/scienceqa/sample_idxs.json', help='path to sample indices file')
    parser.add_argument('--output-dir', type=str, default='../outputs/reasoning', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--num-samples', type=int, default=50, help='# test samples')
    parser.add_argument('--rand-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument('--eg-idx', type=int, default=142, help='index of example sample for few-shot CoT/VQA (default = 142 (CoT), 148 (VQA))')
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print intermediate results')

    # LM params
    parser.add_argument('--temperature', type=float, default=1., help='temperature param for inference')
    parser.add_argument('--max-tokens', type=int, default=100, help='max length (tokens) of generated output')

    # call main with args
    args = parser.parse_args()
    print(args)
    assert args.task in ['cot_zs', 'cot_fs', 'vqa_zs', 'vqa_fs'], 'Invalid task. Please choose from: cot_zs, cot_fs, vqa_zs, vqa_fs'
    main(args)