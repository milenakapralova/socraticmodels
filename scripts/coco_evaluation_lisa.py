'''
This script evaluates the generated captions against the ground truth.

'''

# Package loading
import os
import json
import pandas as pd
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge
import pickle
import numpy as np
import itertools
from bert_score import score

# Local imports
from image_captioning import ClipManager
from utils import get_device


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
        self.imgToEval = {}
        self.res_cossim = res_raw

        #Make res a suitable format for the rule-based evaluation
        res = {}
        for i, row in res_raw.iterrows():
            res[row.image_name] = [{
                'image_id': row.image_name,
                'id': row.image_name,
                'caption': row.generated_caption
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
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts_tok.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts_tok.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

    def evaluate_cossim(self):
        # Get the clip embeddings for images and captions
        with open('data/cache/image_emb.pickle', 'rb') as handle:
            embed_imgs = pickle.load(handle)

        with open('data/cache/embed_capt_gt.pickle', 'rb') as handle:
            embed_capt_gt = pickle.load(handle)

        # Calculate similarities between images and captions
        for img_id in self.intersect_keys:
            # GT
            self.gts_sims[img_id] = (embed_capt_gt[img_id] @ embed_imgs[img_id].T).flatten().tolist()

        gts_list = list(itertools.chain(*self.gts_sims.values()))

        # Calculate aggregates
        self.sims = {
            'gts': [np.mean(gts_list), np.std(gts_list)],
            'res': [self.res_cossim['cosine_similarity'].mean(), self.res_cossim['cosine_similarity'].std()]
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

        P, R, F1 = score(cands, refs, lang="en", verbose=True)

        self.bert_scores = {
            'P': [P.mean(), P.std()],
            'R': [R.mean(), R.std()],
            'F1': [F1.mean(), F1.std()]
        }


def load_result_baseline():
    """
    Load the captions
    """
    try:
        res_baseline = pd.read_csv(f'data/outputs/baseline_outputs.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Either (or both) of the files baseline_outputs.csv and improved_outputs.csv not found! Please run the "
            "coco_captioning_baseline.py or coco_captioning_improved.py to obtain the generated captions before proceeding "
            "with the evaluation."
        )
    return res_baseline


def load_result_improved():
    """
    Load the captions
    """
    try:
        res_improved = pd.read_csv(f'data/outputs/improved_outputs.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Either (or both) of the files baseline_outputs.csv and improved_outputs.csv not found! Please run the "
            "coco_captioning_baseline.py or coco_captioning_improved.py to obtain the generated captions before proceeding "
            "with the evaluation."
        )
    return res_improved


def load_gts_captions():
    # Load the ground truth annotations
    annotation_file = 'data/coco/annotations/captions_val2017.json'

    with open(annotation_file, 'r') as f:
        lines = json.load(f)['annotations']

    gts = {}
    for item in lines:
        if item['image_id'] in gts:
            gts[item['image_id']].append({'image_id': item['image_id'], 'caption': item['caption']})
        else:
            gts[item['image_id']] = [{'image_id': item['image_id'], 'caption': item['caption']}]

    return gts


load_result_baseline()
load_result_improved()
gts = load_gts_captions()


def compute_gts_caption_emb():
    # Compute the embeddings for the gt captions
    file_path = 'data/cache/embed_capt_gt.pickle'
    if not os.path.exists(file_path):

        # Set the device to use
        device = get_device()

        # Instantiate the clip manager
        clip_manager = ClipManager(device)

        embed_capt_gt = {}
        for img_id, list_of_capt_dict in gts.items():
            list_of_captions = [capt_dict['caption'] for capt_dict in list_of_capt_dict]

            # Dims of img_emb_gt: 5 x 768
            img_emb_gt = clip_manager.get_text_emb(list_of_captions)

            embed_capt_gt[img_id] = img_emb_gt

        with open(file_path, 'wb') as handle:
            pickle.dump(embed_capt_gt, handle, protocol=pickle.HIGHEST_PROTOCOL)

compute_gts_caption_emb()

# Evaluation
approaches = ['baseline', 'improved']

data_list = []
rulebased = {}
cossim = {}
bert_score = {}

for approach in approaches:

    # Load the generated captions
    res_raw = pd.read_csv(f'data/outputs/{approach}_outputs.csv')
    res_raw['image_name'] = res_raw['image_name'].str.split('.').str[0].astype(int, copy=False)

    # Instantiate the evaluator
    evaluator = SocraticEvalCap(gts, res_raw)

    # Rule-based metrics
    # evaluator.evaluate_rulebased()
    # for metric, score in evaluator.eval.items():
    #     print(f'{metric}: {score:.3f}')

    # Embedding-based metric
    # evaluator.evaluate_cossim()
    # for source_caption, sim in evaluator.sims.items():
    #     print(f'{source_caption}: avg = {sim[0]:.3f}, std = {sim[1]:.3f}')

    # Learned-based metric
    evaluator.evaluate_bert()
    for metric, score in evaluator.bert_scores.items():
        print(f'{metric}: avg = {score[0]:.3f}, std = {score[1]:.3f}')

    # Store results
    data_list.append({
        **{f'rulebased_{k}': v for k, v in evaluator.eval.items()},
        **{f'sims_{k}': v for k, v in evaluator.sims.items()},
        **{f'bert_{k}': v for k, v in evaluator.bert_scores.items()},
    })

file_path = 'data/outputs/coco_evaluation.csv'
pd.DataFrame(data_list).to_csv(file_path, index=False)