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
from scripts.mm_reasoning import MmReasoner
from utils import print_time_dec, get_samples_sqa, get_device, prepare_dir
import openai

@print_time_dec
def main(
        num_samples, task, eg_idx, lm_model, data_path, random_seed, max_tokens, temperature, output_dir,
        output_file_suffix, verbose
):
    """

    :param num_samples: Number of samples to include in the analysis.
    :param task: Task to run
    :param eg_idx: Index of example sample for few-shot CoT/VQA (default = 142 (CoT), 148 (VQA)).
    :param lm_model: Language model to use.
    :param data_path: Path to sample indices file.
    :param random_seed: Random seed for sampling & inference.
    :param max_tokens: Max length (tokens) of language model generated output.
    :param temperature: Temperature parameter of the language model.
    :param output_dir: Path to output directory.
    :param output_file_suffix: Suffix for output file.
    :param verbose: Whether to print intermediate results
    :return:
    """
    """1. Set up"""
    # Instantiate the multimodal reasoner class.
    mm_reasoner = MmReasoner(random_seed=random_seed)

    # Load sample indices file.
    if not os.path.exists(data_path):
        get_samples_sqa(data_path)
    with open(data_path, 'r') as f:
        sample_idxs_file = json.load(f)
    sample_idxs = sample_idxs_file['cot'] if task in ['cot_zs', 'cot_fs'] else sample_idxs_file['vqa']
    sample_idxs = np.random.choice(sample_idxs, num_samples, replace=False)
    samples = [mm_reasoner.sqa_dataset[int(idx)] for idx in sample_idxs]

    """2. Generate outputs for each sample"""
    outputs = []
    print(f'generating {num_samples} samples for {task} task...')
    for i, sample in tqdm(enumerate(samples)):
        # generate prompt
        # zero-shot CoT
        if task == 'cot_zs':
            prompt = mm_reasoner.create_cot_prompt(sample)
        # few-shot CoT
        elif task == 'cot_fs':
            eg_sample = mm_reasoner.sqa_dataset[eg_idx]
            question1 = mm_reasoner.create_cot_prompt(eg_sample)
            response1 = f'{sample["solution"]}. So the answer is {sample["choices"][sample["answer"]]}\n'
            question2 = mm_reasoner.create_cot_prompt(sample)
            prompt = question1 + response1 + question2
        # VQA
        elif task == 'vqa_zs':
            prompt = mm_reasoner.create_vqa_prompt(sample)
        elif task == 'vqa_fs':
            eg_sample = mm_reasoner.sqa_dataset[eg_idx]
            question1 = mm_reasoner.create_vqa_prompt(eg_sample)
            response1 = f'{sample["answer"]}\n'
            question2 = mm_reasoner.create_vqa_prompt(sample)
            prompt = question1 + response1 + question2
        else:
            raise ValueError(f'Invalid task: {task}. Please choose from: cot_zs, cot_fs, vqa_zs, vqa_fs')

        # generate output
        if lm_model == 'gpt':
            lm_params = {'max_tokens': max_tokens, 'temperature': temperature}
            try:
                output = mm_reasoner.gpt_manager.get_response_gpt(prompt, **lm_params)
            except openai.error.RateLimitError:
                # sleep if API rate limit exceeded
                print('API rate limit exceeded,sleeping for 120s...')
                time.sleep(120)
                output = mm_reasoner.gpt_manager.get_response_gpt(prompt, **lm_params)
        else:
            lm_params = {
                'max_new_tokens': max_tokens, 'temperature': temperature, 'do_sample': False, 'length_penalty': 2.
            }
            output = mm_reasoner.lm_manager.generate_response(prompt, lm_params)
        
        gt = sample['solution'] if task in ['cot_zs', 'cot_fs'] else sample['answer']
        outputs.append({'gt': gt, 'gen': output})
        
        if verbose:
            print(f'{i+1}\nprompt: {prompt}\ngt: {gt}, gen: {output}\n')
        
    print('done.')
    # save results & params
    res_path = f'{output_dir}/{lm_model}/responses_{task}{output_file_suffix}.csv'
    prepare_dir(res_path)
    pd.DataFrame(outputs).to_csv(res_path, index=False)
    print(f'results saved to: {res_path}')
    
if __name__ == '__main__':
    # init argparser
    parser = argparse.ArgumentParser(description='Reasoning')
    # add args
    parser.add_argument('--task', type=str, default='cot_zs', help='task to run')
    parser.add_argument('--lm-model', type=str, default='gpt', help='language model to use')
    parser.add_argument(
        '--data-path', type=str, default='../data/scienceqa/sample_idxs.json', help='path to sample indices file'
    )
    parser.add_argument('--output-dir', type=str, default='../outputs/reasoning', help='path to output directory')
    parser.add_argument('--output-file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--num-samples', type=int, default=50, help='# test samples')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed for sampling & inference')
    parser.add_argument(
        '--eg-idx', type=int, default=142,
        help='index of example sample for few-shot CoT/VQA (default = 142 (CoT), 148 (VQA))'
    )
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print intermediate results')

    # LM params
    parser.add_argument('--temperature', type=float, default=1., help='temperature param for inference')
    parser.add_argument('--max-tokens', type=int, default=100, help='max length (tokens) of generated output')

    # call main with args
    args = parser.parse_args()
    print(args)
    error_message = 'Invalid task. Please choose from: cot_zs, cot_fs, vqa_zs, vqa_fs'
    assert args.task in ['cot_zs', 'cot_fs', 'vqa_zs', 'vqa_fs'], error_message
    main(
        args.num_samples, args.task, args.eg_idx, args.lm_model, args.data_path, args.random_seed, args.max_tokens,
        args.temperature, args.output_dir, args.output_file_suffix, args.verbose
    )