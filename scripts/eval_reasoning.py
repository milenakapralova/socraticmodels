import argparse
import pandas as pd
from evaluate import load
from utils import print_time_dec
import os
import sys
sys.path.append('..')
# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass


def calculate_metrics(gts, gens, metrics):
    """
    Calculate metrics for each sample given ground truth and generated responses.

    :param gts: Ground truths.
    :param gens: Generations.
    :param metrics:
    :return:
    """
    results = {}
    for metric in metrics:
        if metric in ("bleu", "meteor"):
            scorer = load(metric)
            result = scorer.compute(predictions=gens, references=gts)[metric]
        elif metric == "rouge":
            scorer = load(metric)
            result = scorer.compute(predictions=gens, references=gts)['rougeL']
        elif metric == "bertscore":
            scorer = load(metric)
            result = scorer.compute(predictions=gens, references=gts, lang="en")['f1']
        elif metric == "accuracy":
            result = [int(int(gt) == int(gen)) if gen.isdigit() else 0 for gt, gen in zip(gts, gens)]
        else:
            raise ValueError(f'Invalid metric: {metric}. Please choose from: bleu, rouge, meteor, bertscore, accuracy')
        results[metric] = result

    return results

@print_time_dec
def main(task, lm_model, data_dir, file_suffix):
    """
    Runs the multimodel reasoning evaluation.

    :param task: The reasoning task.
    :param lm_model: The language model to use for the reasoning task.
    :param data_dir: The data directory.
    :param file_suffix: The file suffix.
    :return:
    """
    print(f'evaluating {task} task...')
    # load the data
    responses_path = f'{data_dir}/{lm_model}/responses_{task}{file_suffix}.csv'
    data = pd.read_csv(responses_path)
    gts = data['gt'].tolist()
    gens = data['gen'].tolist()
    
    # calculate metrics for each sample
    if task in ('cot_zs',  'cot_fs'):
        metrics = ['bleu', 'rouge', 'meteor', 'bertscore']
    else:
        metrics = ['accuracy']
    
    sample_results = calculate_metrics(gts, gens, metrics)
    # save results
    df_samples = pd.DataFrame(sample_results)
    results = df_samples.describe().loc[['mean', 'std']].transpose()
    results_path = f'{data_dir}/{lm_model}/res_{task}{file_suffix}.csv'
    results.to_csv(results_path, index=True)
    print('done.')
    print(f'results saved to {results_path}')

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate generated responses.')
    parser.add_argument(
        '--data-dir', type=str, default='../outputs/reasoning',
        help='Path to the input dir containing the generated responses.'
    )
    parser.add_argument('--file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--task', type=str, default='cot_zs', help='task to run')
    parser.add_argument('--lm-model', type=str, default='gpt', help='language model to use')
    parser.add_argument('--bleurt-ckpt-path', type=str, default='cache/BLEURT-20-D6', help='path to bleurt checkpoint')
    args = parser.parse_args()
    error_message = f'Invalid task: {args.task}. Please choose from: cot_zs, cot_fs, vqa_zs, vqa_fs'
    assert args.task in ['cot_zs', 'cot_fs', 'vqa_zs', 'vqa_fs'], error_message
    print(args)
    main(args.task, args.lm_model, args.data_dir, args.file_suffix)
