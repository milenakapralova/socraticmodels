# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import argparse
from datasets import load_dataset
import image_captioning as ic
from utils import print_time_dec

@print_time_dec
def main(args):
    '''1. Set up'''
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # instantiate managers
    clip_manager = ic.ClipManager(device)
    vocab_manager = ic.VocabManager()
    cache_manager = ic.CacheManager()
    # instantiate prompt generator
    prompt_generator = ic.LmPromptGenerator()
    
    # compute place & object features
    place_feats = cache_manager.get_place_emb(clip_manager, vocab_manager)
    obj_feats = cache_manager.get_object_emb(clip_manager, vocab_manager)
    
    # load dataset
    # load scienceQA dataset
    scienceQA_dataset = load_dataset('derek-thomas/ScienceQA', split='validation')
    # filter out samples with no image
    scienceQA_dataset = [sample for sample in scienceQA_dataset if sample['image'] is not None]
    #TODO: take random or specific samples?
    np.random.seed(args.rand_seed)
    sample_idxs = np.random.choice(len(scienceQA_dataset), args.num_samples, replace=False)
    samples = [scienceQA_dataset[idx] for idx in sample_idxs]

    '''2. Generate outputs for each sample'''
    outputs = []
    print(f'generating {args.num_samples} samples for {args.task} task...')
    for sample in tqdm(samples):
        # generate prompt
        # zero-shot CoT
        if args.task == 'cot_zs':
            prompt = prompt_generator.create_cot_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        # few-shot CoT
        elif args.task == 'cot_fs':
            target_sample = scienceQA_dataset[args.target_idx]
            prompt = prompt_generator.create_cot_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats) + f'{sample["solution"]}. So the answer is {sample["choices"][sample["answer"]]}\n' + prompt_generator.create_cot_prompt(target_sample, clip_manager, vocab_manager, place_feats, obj_feats)
        # VQA
        elif args.task == 'vqa':
            prompt = prompt_generator.create_vqa_prompt(sample, clip_manager, vocab_manager, place_feats, obj_feats)
        else:
            raise ValueError(f'Invalid task: {args.task}. Please choose from: cot-zs, cot-fs, vqa')

        if args.verbose:
            print(f'prompt: {prompt}')
        # generate output
        output = ic.get_response_gpt(prompt, args.model, args.temperature, args.max_tokens)
        gt = sample['solution'] if args.task == 'cot_zs' or args.task == 'cot_fs' else sample['answer'] if args.task == 'vqa' else None
        outputs.append({
                'gt': gt,
                'gen': output
            })
        
    print('done.')
    # save results & params
    res_dir = args.output_dir
    os.makedirs(res_dir, exist_ok=True)
    res_path = f'{res_dir}/responses_{args.task}{args.output_file_suffix}.csv'
    pd.DataFrame(outputs).to_csv(res_path, index=False)
    print(f'results saved to: {res_path}')
    
if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Reasoning')
    # add args
    parser.add_argument('--task', type=str, default='cot_zs', help='task to run')
    parser.add_argument('--output-dir', type=str, default='outputs/reasoning', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--num-samples', type=int, default=100, help='# test samples')
    parser.add_argument('--rand-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print intermediate results')

    # GPT-3 params
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='gpt3 model version')
    parser.add_argument('--temperature', type=float, default=1., help='temperature param for inference')
    parser.add_argument('--max-tokens', type=int, default=100, help='max length (tokens) of generated output')

    # call main with args
    args = parser.parse_args()
    print(args)
    main(args)