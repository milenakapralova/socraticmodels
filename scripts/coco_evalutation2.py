#!/usr/bin/env python
# coding: utf-8

# # SocraticFlanT5 - Evaluation | DL2 Project, May 2023
# ---
# 
# This notebook evaluates the generated captions based on the MS COCO ground-truth captions. We will evaluate the folowing two approaches: 
# 1. <span style="color:#006400">**Baseline**</span>: a Socratic model based on the work by [Zeng et al. (2022)](https://socraticmodels.github.io/) where GPT-3 is replaced by [FLAN-T5-xl](https://huggingface.co/docs/transformers/model_doc/flan-t5). 
# 
# 2. <span style="color:#006400">**Improved prompting**</span>: an improved baseline model where the template prompt filled by CLIP is processed before passing to FLAN-T5-xl.
# 
# There are two approaches to this evaluation: rule-based and embedding-based.
# 
# ---
# For the **rule-based approach**, the following metrics will be used, based on [this](https://github.com/salaniz/pycocoevalcap) repository:
# 
# * *BLEU-4*: BLEU (Bilingual Evaluation Understudy) is a metric that measures the similarity between the generated captions and the ground truth captions based on n-gram matching. The BLEU-4 score measures the precision of the generated captions up to four-grams compared to the ground truth captions.
# 
# * *METEOR*: METEOR (Metric for Evaluation of Translation with Explicit ORdering) is another metric that measures the similarity between the generated captions and the ground truth captions. It also takes into account word order and synonymy by using a set of reference summaries to compute a harmonic mean of precision and recall.
# 
# * *CIDEr*: CIDEr (Consensus-based Image Description Evaluation) is a metric that measures the consensus between the generated captions and the ground truth captions. It computes the similarity between the generated captions and the reference captions based on their TF-IDF weights, which helps capture important words in the captions.
# 
# * *SPICE*: SPICE (Semantic Propositional Image Caption Evaluation) is a metric that measures the semantic similarity between the generated captions and the ground truth captions. It analyzes the similarity between the semantic propositions present in the generated captions and those in the reference captions, taking into account the structure and meaning of the propositions.
# 
# * *ROUGE-L*: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a metric that measures the similarity between the generated captions and the ground truth captions based on overlapping sequences of words. ROUGE-L measures the longest common subsequence (LCS) between the generated captions and the reference captions, taking into account sentence-level structure and word order.
# 
# ---
# 
# For the **embedding-based** approach (based on CLIP embeddings), we calculate the cosine similarities between each image embedding and embeddings of the ground truth captions and then we calculate the cosine similarities between each image embedding and embeddings of the captions generated with FLAN-T5-xl.
# 
# ---

# <span style="color:#8B0000">**Important**</span>: we assume that you have the generated captions accessible from the current directory via `cache/baseline_outputs.csv` or `cache/improved_outputs.csv` or both. If that is not the case, please run (either or both of) the following notebooks:
# * `coco_captioning_baseline.ipynb`
# * `coco_captioning_improved.ipynb`
# 
# Moreover, we assume you have pre-computed the image embeddings and have them stored and accessible from the current directory via `cache/embed_imgs.pickle`. If that is not the case, please run (either or both of) the following notebook:
# * `coco_image_embedding.ipynb`

# ### Set-up

# #### Loading the required packages

# In[1]:


# Package loading
import os
import json
import pickle
import pandas as pd

# Local imports
from image_captioning import ClipManager
from coco_evaluation import SocraticEvalCap
from utils import get_device, prepare_dir

# ### Evaluate the generated captions against the ground truth

# #### Load the ground truth annotations

# In[2]:

# Assessing whether the generated data is present so that we have something to evaluate

try:
    res_baseline = pd.read_csv(f'../data/outputs/baseline_outputs.csv')
    res_improved = pd.read_csv(f'../data/outputs/improved_outputs.csv')

except FileNotFoundError:
    print(
        "Either (or both) of the files baseline_outputs.csv and improved_outputs.csv not found! Please run the "
        "coco_captioning_baseline.py or coco_captioning_improved.py to obtain the generated captions before proceeding "
        "with the evaluation."
    )
    raise



annotation_file = '../data/coco/annotations/captions_val2017.json'

with open(annotation_file, 'r') as f:
    lines = json.load(f)['annotations']
gts = {}
for item in lines:
    if item['image_id'] not in gts:
        gts[item['image_id']] = []
    gts[item['image_id']].append({'image_id': item['image_id'], 'caption': item['caption']})


# #### Compute the embeddings for the gt captions

# In[3]:

file_path = '../data/cache/embed_capt_gt.pickle'
if not os.path.exists(file_path):
    prepare_dir(file_path)

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


# #### Evaluation

# In[4]:


approaches = ['baseline', 'improved']
# approaches = ['baseline']


eval_cap = {
    'rulebased': {},
    'cossim': {},
    'bert_score': {}
}


for approach in approaches:
    # Load the generated captions
    res_raw = pd.read_csv(f'../data/outputs/{approach}_outputs.csv')
    res_raw['image_path'] = res_raw['image_path'].str.split('.').str[0].astype(int, inplace=True)


    evaluator = SocraticEvalCap(gts, res_raw)

    # Rule-based metrics
    evaluator.evaluate_rulebased()
    eval_rulebased = {}

    for metric, score in evaluator.eval.items():
        print(f'{metric}: {score:.3f}')
        eval_rulebased[metric] = round(score, 5)
    eval_cap['rulebased'][approach] = eval_rulebased

    # Embedding-based metric
    evaluator.evaluate_cossim()
    for source_caption, sim in evaluator.sims.items():
        print(f'{source_caption}: avg = {sim[0]:.3f}, std = {sim[1]:.3f}')
    eval_cap['cossim'][approach] = evaluator.sims

    # Learned-based metric
    evaluator.evaluate_bert()
    for metric, score in evaluator.bert_scores.items():
        print(f'{metric}: avg = {score[0]:.3f}, std = {score[1]:.3f}')
    eval_cap['bert_score'][approach] = evaluator.bert_scores

# ### Save the outputs

# In[5]:


file_path = '../data/outputs/eval_cap.pickle'
prepare_dir(file_path)
with open(file_path, 'wb') as handle:
    pickle.dump(eval_cap, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




