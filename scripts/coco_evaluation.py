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
import uuid



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

    def evaluate_cossim(self, gt_caption_emb, image_emb):

        # Calculate similarities between images and captions
        for img_id in self.intersect_keys:
            # GT
            self.gts_sims[img_id] = (gt_caption_emb[img_id] @ image_emb[img_id].T).flatten().tolist()

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

        p, r, f1 = score(cands, refs, lang="en", verbose=True)

        self.bert_scores = {
            'p': p,
            'r': r,
            'f1': f1
        }


def load_result_baseline():
    """
    Load the captions
    """
    try:
        res_baseline = pd.read_csv(f'../data/outputs/baseline_outputs.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "baseline_outputs.csv not found! Please run the coco_captioning_baseline.py or coco_captioning_improved.py "
            "to obtain the generated captions before proceeding with the evaluation."
        )
    return res_baseline


def load_result_improved():
    """
    Load the captions
    """
    try:
        res_improved = pd.read_csv(f'../data/outputs/improved_outputs.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "improved_outputs.csv not found! Please run the coco_captioning_baseline.py or coco_captioning_improved.py "
            "to obtain the generated captions before proceeding with the evaluation."
        )
    return res_improved


# Package loading
import os
import json
import pickle
import pandas as pd

try:
    os.chdir('scripts')
except:
    pass


# Local imports
from scripts.image_captioning import ClipManager, ImageManager, CocoManager
from scripts.utils import get_device, prepare_dir

def load_gts_captions():
    # Load the ground truth annotations
    annotation_file = '../data/coco/annotations/captions_val2017.json'

    with open(annotation_file, 'r') as f:
        lines = json.load(f)['annotations']

    gts = {}
    for item in lines:
        if item['image_id'] in gts:
            gts[item['image_id']].append({'image_id': item['image_id'], 'caption': item['caption']})
        else:
            gts[item['image_id']] = [{'image_id': item['image_id'], 'caption': item['caption']}]

    return gts


def get_img_list_sorted(res_baseline):
    img_list = res_baseline['image_name'].tolist()
    img_list.sort()
    return img_list


def get_uuid_for_imgs(img_list):
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(img_list)))


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

        # Set the image folder path
        img_folder = '../data/coco/val2017/'

        # Instantiate the image manager
        image_manager = ImageManager()

        image_emb = {}
        for img_name in img_list:
            img = image_manager.load_image(img_folder + img_name)
            img_emb = clip_manager.get_img_emb(img)
            image_emb[int(img_name.split('.')[0])] = img_emb.flatten()

        with open(emb_pickle_path, 'wb') as handle:
            pickle.dump(image_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return image_emb


def evaluate_captions(data_to_analyse, gt_caption_emb, image_emb):
    # Create a data list to store the outputs
    data_list = []

    for approach, data_df in data_to_analyse.items():

        data_df['image_name'] = data_df['image_name'].map(lambda x: int(x.split('.')[0]))

        # Instantiate the evaluator
        evaluator = SocraticEvalCap(gts, data_df)

        # Rule-based metrics
        evaluator.evaluate_rulebased()
        # for metric, score in evaluator.eval.items():
        #     print(f'{metric}: {score:.3f}')

        # Embedding-based metric
        evaluator.evaluate_cossim(gt_caption_emb, image_emb)
        # for source_caption, sim in evaluator.sims.items():
        #     print(f'{source_caption}: avg = {sim[0]:.3f}, std = {sim[1]:.3f}')

        # Learned-based metric
        evaluator.evaluate_bert()
        # for metric, score in evaluator.bert_scores.items():
        #     print(f'{metric}: avg = {score[0]:.3f}, std = {score[1]:.3f}')

        # Store results
        for i, data_dic in enumerate(evaluator.evalImgs):
            data_list.append({
                'approach': approach,
                'caption': evaluator.res[data_dic['image_id']][0]['caption'],
                **data_dic,
                'gts_sims': evaluator.gts_sims[data_dic['image_id']],
                'bert_p': float(evaluator.bert_scores['p'][i]),
                'bert_r': float(evaluator.bert_scores['r'][i]),
                'bert_f1': float(evaluator.bert_scores['f1'][i]),
            })
    return pd.DataFrame(data_list)


# Load the results
res_baseline = load_result_baseline()
res_improved = load_result_improved()

# Load the captions
gts = load_gts_captions()

# Extract a sorted list of the images whose captions will be evaluated
img_list = get_img_list_sorted(res_baseline)

# Set the device to use
device = get_device()

# Instantiate the clip manager
clip_manager = ClipManager(device)

# Retrieve the embeddings of the ground truth captions
gt_caption_emb = load_caption_emb(clip_manager, gts, img_list)

# Retrieve the embeddings of the images
image_emb = load_image_emb(clip_manager, img_list)

data_to_analyse = {
    'baseline': res_baseline,
    'improved': res_improved
}

analysis_df = evaluate_captions(data_to_analyse, gt_caption_emb, image_emb)
analysis_df.to_csv('../data/outputs/caption_eval.csv', index=False)



