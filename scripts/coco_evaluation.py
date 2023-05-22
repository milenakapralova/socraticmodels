# Package loading
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge
import numpy as np
import itertools
from bert_score import score
import os
import sys
import json
import pickle
import pandas as pd
import argparse
sys.path.append('..')
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.image_captioning import ClipManager, ImageManager
from scripts.utils import get_device, prepare_dir, get_uuid_for_imgs


class SocraticEvalCap:
    def __init__(self, gts, res_raw):
        """
        Adapted from the COCOEvalCap class from pycocoevalcap/coco_evaluation.py.

        :param coco:
        :param cocoRes:
        """
        self.evalImgs = []
        self.eval = {}
        self.gts_sims = {}
        self.res_sims = {}
        self.img_to_eval = {}
        self.res_cossim = res_raw
        self.res_cossim_map = dict(zip(res_raw['image_id'], res_raw['cos_sim']))

        #Make res a suitable format for the rule-based evaluation
        res = {}
        for i, row in res_raw.iterrows():
            res[row.image_id] = [{
                'image_id': row.image_id,
                'id': row.image_id,
                'caption': row.best_caption
            }]

        self.intersect_keys = set(gts.keys()) & set(res.keys())
        gts = {key: gts.get(key) for key in self.intersect_keys}
        res = {key: res.get(key) for key in self.intersect_keys}

        self.res = res
        self.gts = gts
        self.img_ids = self.gts.keys()

    def evaluate_rulebased(self):
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts_tok = tokenizer.tokenize(self.gts)
        res_tok = tokenizer.tokenize(self.res)
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts_tok, res_tok)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_imgs_to_eval(scs, gts_tok.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.set_eval(score, method)
                self.set_imgs_to_eval(scores, gts_tok.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_imgs_to_eval(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.img_to_eval:
                self.img_to_eval[imgId] = {}
                self.img_to_eval[imgId]["image_id"] = imgId
            self.img_to_eval[imgId][method] = score

    def set_eval_imgs(self):
        self.evalImgs = [eval for imgId, eval in self.img_to_eval.items()]

    def evaluate_cossim(self, gt_caption_emb, image_emb):

        # Calculate similarities between images and captions
        for img_id in self.intersect_keys:
            # GT
            self.gts_sims[img_id] = (gt_caption_emb[img_id] @ image_emb[img_id].T).flatten().tolist()

        gts_list = list(itertools.chain(*self.gts_sims.values()))

        # Calculate aggregates
        self.sims = {
            'gts': [np.mean(gts_list), np.std(gts_list)],
            'res': [self.res_cossim['cos_sim'].mean(), self.res_cossim['cos_sim'].std()]
        }

    def evaluate_bert(self):
        cands = []
        for image_id, caption_list in self.res.items():
            cpt_dict = caption_list[0]
            cands.append(cpt_dict['caption'])

        refs = []
        for image_id, caption_list in self.gts.items():
            current_ref = ''
            for cpt_dict in caption_list:
                current_ref += ' ' + cpt_dict['caption']
            refs.append(current_ref)

        p, r, f1 = score(cands, refs, lang="en", verbose=True)

        self.bert_scores = {
            'p': p,
            'r': r,
            'f1': f1
        }


def load_caption_baseline(captions_path):
    """
    Load the captions
    """
    try:
        res_baseline = pd.read_csv(captions_path)
    except FileNotFoundError:
            "Baseline captions csv not found! Make sure you pass the correct path as an arg. Please run coco_captioning_baseline.py to obtain the generated captions before proceeding with the evaluation."
    return res_baseline


def load_caption_improved(captions_path):
    """
    Load the captions
    """
    try:
        res_improved = pd.read_csv(captions_path)
    except FileNotFoundError:
            "Improved captions csv not found! Make sure you pass the correct path as an arg. Please run coco_captioning_improved.py to obtain the generated captions before proceeding with the evaluation."
    return res_improved


def load_gts_captions(annot_path):
    # Load the ground truth annotations

    with open(annot_path, 'r') as f:
        lines = json.load(f)['annotations']

    gts = {}
    for item in lines:
        if item['image_id'] in gts:
            gts[item['image_id']].append({'image_id': item['image_id'], 'caption': item['caption']})
        else:
            gts[item['image_id']] = [{'image_id': item['image_id'], 'caption': item['caption']}]

    return gts


def load_caption_emb(clip_manager, gts, img_list):

    emb_pickle_path = f'../data/cache/caption_emb_{get_uuid_for_imgs(img_list)}.pickle'

    if os.path.exists(emb_pickle_path):
        with open(emb_pickle_path, 'rb') as f:
            gt_caption_emb = pickle.load(f)
    else:
        # Ensure the directory exists
        prepare_dir(emb_pickle_path)

        gt_caption_emb = {}
        for img_name in img_list:
            img_id = int(img_name.split('.')[0])
            list_of_captions = [capt_dict['caption'] for capt_dict in gts[img_id]]

            # Dims of img_emb_gt: 5 x 768
            img_emb_gt = clip_manager.get_text_emb(list_of_captions)

            gt_caption_emb[img_id] = img_emb_gt

        with open(emb_pickle_path, 'wb') as handle:
            pickle.dump(gt_caption_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return gt_caption_emb


def load_image_emb(clip_manager, img_list):

    emb_pickle_path = f'../data/cache/img_emb_{get_uuid_for_imgs(img_list)}.pickle'

    if os.path.exists(emb_pickle_path):
        with open(emb_pickle_path, 'rb') as f:
            image_emb = pickle.load(f)
    else:
        # Ensure the directory exists
        prepare_dir(emb_pickle_path)

        # Instantiate the image manager
        image_manager = ImageManager()

        image_emb = {}
        for img_name in img_list:
            img = image_manager.load_image(f'{args.img_dir}/{img_name}')
            img_emb = clip_manager.get_img_emb(img)
            image_emb[int(img_name.split('.')[0])] = img_emb.flatten()

        with open(emb_pickle_path, 'wb') as handle:
            pickle.dump(image_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return image_emb


def evaluate_captions(data_to_analyse, gts, gt_caption_emb, image_emb):
    # Create a data list to store the outputs
    data_list = []

    for approach, data_df in data_to_analyse.items():

        data_df['image_id'] = data_df['image_name'].map(lambda x: int(x.split('.')[0]))

        # Instantiate the evaluator
        evaluator = SocraticEvalCap(gts, data_df)

        # Rule-based metrics
        evaluator.evaluate_rulebased()

        # Embedding-based metric
        evaluator.evaluate_cossim(gt_caption_emb, image_emb)

        # Learned-based metric
        evaluator.evaluate_bert()

        # Store evaluations
        for i, data_dic in enumerate(evaluator.evalImgs):
            data_list.append({
                'approach': approach,
                'caption': evaluator.res[data_dic['image_id']][0]['caption'],
                **{k: v for k, v in data_dic.items()},
                'SPICE': data_dic['SPICE']['All']['f'],
                'gts_sims': evaluator.gts_sims[data_dic['image_id']],
                'bert_p': float(evaluator.bert_scores['p'][i]),
                'bert_r': float(evaluator.bert_scores['r'][i]),
                'bert_f1': float(evaluator.bert_scores['f1'][i]),
                'cossim': evaluator.res_cossim_map[data_dic['image_id']]
            })
    return pd.DataFrame(data_list)

def summarise_analysis(analysis_df):
    analysis_df['gts_sims'] = analysis_df['gts_sims'].map(lambda x: np.mean(x))
    numerical_cols = [c for c in analysis_df.columns if c not in ('approach', 'caption')]
    return analysis_df.groupby('approach')[numerical_cols].mean().reset_index()

def main(args):
    print('computing metrics for baseline & improved captions...')
    # Load the generated captions
    caption_baseline = load_caption_baseline(args.baseline_captions_path)
    caption_improved = load_caption_improved(args.improved_captions_path)

    # Load the ground truth captions
    gts = load_gts_captions(args.annot_path)

    # Extract the list of images
    img_list = caption_baseline['image_name'].tolist()

    # Set the device to use
    device = get_device()

    # Instantiate the clip manager
    clip_manager = ClipManager(device)

    # Retrieve the embeddings of the ground truth captions
    gt_caption_emb = load_caption_emb(clip_manager, gts, img_list)

    # Retrieve the embeddings of the images
    image_emb = load_image_emb(clip_manager, img_list)

    # Perform analysis
    data_to_analyse = {
        'baseline': caption_baseline,
        'improved': caption_improved
    }
    analysis_df = evaluate_captions(data_to_analyse, gts,  gt_caption_emb, image_emb)
    analysis_df_gr = summarise_analysis(analysis_df)

    # Output analysis
    os.makedirs(args.output_dir, exist_ok=True)
    analysis_df.to_csv(f'{args.output_dir}/captions_eval.csv', index=False)
    analysis_df_gr.to_csv(f'{args.output_dir}/captions_eval_summary.csv', index=False)
    
    print('done')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='args for eval of baseline & improved captioning on COCO')
    parser.add_argument('--img-dir', type=str, default='../data/coco/val2017', help='path to image directory')
    parser.add_argument('--annot-path', type=str, default='../data/coco/annotations/captions_val2017.json', help='path to annotations file')
    parser.add_argument('--baseline-captions-path', type=str, default='../outputs/captions/google/flan-t5-xl/baseline_captions.csv', help='path to baseline captions file')
    parser.add_argument('--improved-captions-path', type=str, default='../outputs/captions/google/flan-t5-xl/improved_captions.csv', help='path to improved captions file')
    parser.add_argument('--output-dir', type=str, default='../outputs/captions/google/flan-t5-xl', help='path to output directory')

    args = parser.parse_args()
    print(args)
    
    main(args)
    