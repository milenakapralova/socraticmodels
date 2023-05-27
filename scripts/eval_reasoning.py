import argparse
import pandas as pd
from evaluate import load
from utils import print_time_dec

def calculate_metrics(gts, gens, metrics):
    '''Calculate metrics for each sample given ground truth and generated responses.'''
    results = {}
    for metric in metrics:
        scorer = load(metric)
        if metric == "bleu" or metric == "meteor":
            result = scorer.compute(predictions=gens, references=gts)[metric]
        elif metric == "rouge":
            result = scorer.compute(predictions=gens, references=gts)['rougeL']
        elif metric == "bleurt":
            scorer = load('bleurt', 'bleurt-base-128')
            result = scorer.compute(predictions=gens, references=gts)['scores']
        elif metric == "bertscore":
            result = scorer.compute(predictions=gens, references=gts, lang="en")['f1']
        elif metric == "accuracy":
            result = sum([int(gt.lower() == gen.lower()) for gt, gen in zip(gts, gens)])/len(gts)
        else:
            raise ValueError(f'Invalid metric: {metric}. Please choose from: bleu, rouge, meteor, bertscore, bleurt')
        results[metric] = result

    return results

@print_time_dec
def main(args):
    print(f'evaluating {args.task} task...')
    # load the data
    responses_path = f'{args.data_dir}/responses_{args.task}{args.file_suffix}.csv'
    data = pd.read_csv(responses_path)
    gts = data['gt'].tolist()
    gens = data['gen'].tolist()
    
    # calculate metrics for each sample
    if args.task == 'cot_zs' or args.task == 'cot_fs':
        metrics = ['bleu', 'rouge', 'meteor', 'bertscore', 'bleurt']
    elif args.task == 'vqa':
        metrics = ['accuracy']
    else:
        raise ValueError(f'Invalid task: {args.task}. Please choose from: cot-zs, cot-fs, vqa')
    
    sample_results = calculate_metrics(gts, gens, metrics)
    # save results
    df_samples = pd.DataFrame(sample_results)
    results = df_samples.describe().loc[['mean', 'std']].transpose()
    results_path = f'{args.data_dir}/{args.lm_model}/res_{args.task}{args.file_suffix}.csv'
    results.to_csv(results_path, index=True)
    print('done.')
    print(f'results saved to {results_path}')

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate generated responses.')
    parser.add_argument('--data-dir', type=str, default='outputs/reasoning', help='Path to the input dir containing the generated responses.')
    parser.add_argument('--file-suffix',  type=str, default='', help='suffix for output file')
    parser.add_argument('--task', type=str, default='cot_zs', help='task to run')
    parser.add_argument('--lm-model', type=str, default=None, help='language model to use')
    args = parser.parse_args()
    print(args)
    main(args)
