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
import json
import pickle
import pandas as pd
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
        self.imgToEval = {}
        self.res_cossim = res_raw
        self.res_cossim_map = dict(zip(res_raw['image_id'], res_raw['cosine_similarity']))

        #Make res a suitable format for the rule-based evaluation
        res = {}
        for i, row in res_raw.iterrows():
            res[row.image_id] = [{
                'image_id': row.image_id,
                'id': row.image_id,
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


def load_caption(caption):
    """
    Loads a caption csv file from the csv name passed.
    """
    try:
        caption_df = pd.read_csv(caption)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{caption} not found! Please run the coco_captioning_baseline.py and coco_captioning_improved.py "
            "to obtain the generated captions before proceeding with the evaluation."
        )
    return caption_df


def load_all_captions():
    """
    Load all the captions in the '../data/outputs/captions/' directory.

    :return: Dictionary mapping caption csv file names to the loaded dataframe.
    """
    caption_dir = '../data/outputs/captions/'
    return {c.split('.')[0]: load_caption(caption_dir + c) for c in os.listdir(caption_dir) if c.endswith('csv')}


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


def perform_agg(analysis_df, numerical_cols, agg_type):
    """
    Performs a group by operation on analysis df.

    :param analysis_df: The dataframe to group.
    :param numerical_cols: The numerical columns to be grouped.
    :param agg_type: The aggregation operator as a string.
    :return:
    """
    # Define a function map for the aggregation
    agg_func_map = {
        'mean': np.mean,
        'std': np.std,
    }
    # Calculate the ground truth column aggregation
    gts_sims_map = {}
    for approach in analysis_df['approach'].unique():
        temp_df = analysis_df[analysis_df['approach'] == approach]
        temp_gts = np.concatenate(temp_df['gts_sims'].map(lambda x: np.array(x)).tolist())
        gts_sims_map[approach] = agg_func_map[agg_type](temp_gts)
    # Aggregate the other columns
    agg_df = analysis_df.groupby('approach').agg({c: agg_func_map[agg_type] for c in numerical_cols})
    agg_df.columns = [f'{c}_{agg_type}' for c in agg_df.columns]
    # Reset the index
    agg_df = agg_df.reset_index()
    # Set the ground truth value
    agg_df[f'gts_sims_{agg_type}'] = agg_df['approach'].map(gts_sims_map)
    # Return the dataframe
    return agg_df.reset_index()


def summarise_analysis(analysis_df):
    numerical_cols = [c for c in analysis_df.columns if c not in ('approach', 'caption', 'image_id', 'gts_sims')]
    mean_df = perform_agg(analysis_df, numerical_cols, agg_type='mean')
    std_df = perform_agg(analysis_df, numerical_cols, agg_type='std')
    return pd.concat([mean_df, std_df], axis=1)


# Load the generated captions
caption_dic = load_all_captions()

# Load the ground truth captions
gts = load_gts_captions()

# Extract the list of images
img_list = list(caption_dic.values())[0]['image_name'].tolist()

# Set the device to use
device = get_device()

# Instantiate the clip manager
clip_manager = ClipManager(device)

# Retrieve the embeddings of the ground truth captions
gt_caption_emb = load_caption_emb(clip_manager, gts, img_list)

# Retrieve the embeddings of the images
image_emb = load_image_emb(clip_manager, img_list)

analysis_df = evaluate_captions(caption_dic, gt_caption_emb, image_emb)
analysis_df_gr = summarise_analysis(analysis_df)

# Prepare output folder
out_folder = '../data/outputs/analysis/'
prepare_dir(out_folder)

# Output analysis
analysis_df.to_csv(out_folder + 'caption_eval.csv', index=False)
analysis_df_gr.to_csv(out_folder + 'caption_eval_summary.csv', index=False)

