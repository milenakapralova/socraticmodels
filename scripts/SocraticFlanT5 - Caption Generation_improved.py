#!/usr/bin/env python
# coding: utf-8

# # SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
# ---
# 
# This notebook downloads the images from the validation split of the [MS COCO Dataset (2017 version)](https://cocodataset.org/#download) and the corresponding ground-truth captions and generates captions based on the Socratic model pipeline outlined below. In this notebook, we propose a new method to obtain image captions via the Socratic method:
# * Improved prompting: an improved baseline model where the template prompt filled by CLIP is processed before passing to [FLAN-T5-xl](https://huggingface.co/docs/transformers/model_doc/flan-t5).
# 
# In other words, this is an improved pipeline that has for goal to generate similar or improved captions using open-source and free models.

# ## Set-up
# If you haven't done so already, please activate the corresponding environment by running in the terminal: `conda env create -f environment.yml`. Then type `conda activate socratic`.

# ### Loading the required packages

# In[4]:


# Package loading
import matplotlib.pyplot as plt
import pandas as pd
from transformers import set_seed
import os
import numpy as np
import re
import pickle
import time
import random
import pandas as pd

# Local imports
from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, COCOManager
from image_captioning import LmPromptGenerator as pg
from utils import get_device


# ### Set seeds for reproducible results

# In[2]:


# Set HuggingFace seed
set_seed(42)

# Set seed for 100 random images of the MS COCO validation split
random.seed(42)


# ## Step 1: Downloading the MS COCO images and annotations

# In[5]:


imgs_folder = 'imgs/val2017/'
annotation_file = '../annotations/annotations/captions_val2017.json'

coco_manager = COCOManager()
coco_manager.download_data()


# ## Step 2: Generating the captions via the Socratic pipeline
# 

# ### Set the device and instantiate managers

# In[7]:


# Set the device to use
device = get_device()

# Instantiate the clip manager
clip_manager = ClipManager(device)

# Instantiate the image manager
image_manager = ImageManager()

# Instantiate the vocab manager
vocab_manager = VocabManager()

# Instantiate the Flan T5 manager
flan_manager = FlanT5Manager()


# ### Compute place and object features

# In[8]:


# Calculate the place features
if not os.path.exists('../cache/place_feats.npy'):
    # Calculate the place features
    place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])
    np.save('../cache/place_feats.npy', place_feats)
else:
    place_feats = np.load('../cache/place_feats.npy')

# Calculate the object features
if not os.path.exists('../cache/object_feats.npy'):
    # Calculate the object features
    object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])
    np.save('../cache/object_feats.npy', object_feats)
else:
    object_feats = np.load('../cache/object_feats.npy')


# ### Load images and compute image embedding

# In[9]:


approach = 'improved'

img_dic = {}
img_feat_dic = {}
img_paths = {}
# if not os.path.exists(f'{approach}_outputs.csv'):
    # N = len(os.listdir(imgs_folder))
N = 100
random_numbers = random.sample(range(len(os.listdir(imgs_folder))), N)

# for ix, file_name in enumerate(os.listdir(imgs_folder)[:N]):
for ix, file_name in enumerate(os.listdir(imgs_folder)):
     # Consider only image files that are part of the random sample
    if file_name.endswith(".jpg") and ix in random_numbers:  
        # Getting image id
        file_name_strip = file_name.strip('.jpg')
        match = re.search('^0+', file_name_strip)
        sequence = match.group(0)
        image_id = int(file_name_strip[len(sequence):])

        img_path = os.path.join(imgs_folder, file_name)
        img = image_manager.load_image(img_path)
        img_feats = clip_manager.get_img_feats(img)
        img_feats = img_feats.flatten()
        img_paths[image_id] = file_name

        img_dic[image_id] = img
        img_feat_dic[image_id] = img_feats


# ### Zero-shot VLM (CLIP)
# We zero-shot prompt CLIP to produce various inferences of an iage, such as image type or the number of people in an image:

# #### Classify image type

# In[10]:


img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_feats = clip_manager.get_text_feats([f'This is a {t}.' for t in img_types])

# Create a dictionary to store the image types
img_type_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_img_types, img_type_scores = clip_manager.get_nn_text(img_types, img_types_feats, img_feat)
    img_type_dic[img_name] = sorted_img_types[0]


# #### Classify number of people

# In[ ]:


ppl_texts = [
    'are no people', 'is one person', 'are two people', 'are three people', 'are several people', 'are many people'
]
ppl_feats = clip_manager.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])

# Create a dictionary to store the number of people
num_people_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_feats, img_feat)
    num_people_dic[img_name] = sorted_ppl_texts[0]


# #### Classify image place

# In[ ]:


place_topk = 3

# Create a dictionary to store the number of people
location_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_feats, img_feat)
    location_dic[img_name] = sorted_places[0]


# #### Classify image object

# In[ ]:


obj_topk = 10

# Create a dictionary to store the similarity of each object with the images
object_score_map = {}
sorted_obj_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_feats, img_feat)
    object_score_map[img_name] = dict(zip(sorted_obj_texts, obj_scores))
    sorted_obj_dic[img_name] = sorted_obj_texts
    object_list = ''
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
    object_list = object_list[:-2]


# #### Finding both relevant and different objects using cosine similarity

# In[ ]:


# Create a dictionary that maps the objects to the cosine sim.
object_embeddings = dict(zip(vocab_manager.object_list, object_feats))

# Create a dictionary to store the terms to include
terms_to_include = {}

for img_name, sorted_obj_texts in sorted_obj_dic.items():

    # Create a list that contains the objects ordered by cosine sim.
    embeddings_sorted = [object_embeddings[w] for w in sorted_obj_texts]

    # Create a list to store the best matches
    best_matches = [sorted_obj_texts[0]]

    # Create an array to store the embeddings of the best matches
    unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

    # Loop through the 100 best objects by cosine similarity
    for i in range(1, 100):
        # Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
        max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
        # If object i is different enough to the current best matches, add it to the best matches
        if max_cos_sim < 0.7:
            unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
            best_matches.append(sorted_obj_texts[i])

    # Looping through the best matches, consider each terms separately by splitting the commas and spaces.
    data_list = []
    for terms in best_matches:
        for term_split in terms.split(', '):
            score = clip_manager.get_image_caption_score(term_split, img_feat_dic[img_name])
            data_list.append({
                'term': term_split, 'score': score, 'context': terms
            })
            term_split_split = term_split.split(' ')
            if len(term_split_split) > 1:
                for term_split2 in term_split_split:
                    score = clip_manager.get_image_caption_score(term_split2, img_feat_dic[img_name])
                    data_list.append({
                        'term': term_split2, 'score': score, 'context': terms
                    })

    # Create a dataframe with the terms and scores and only keep the top term per context.
    term_df = pd.DataFrame(data_list).sort_values('score', ascending=False).drop_duplicates('context').reset_index(drop=True)

    # Prepare loop to find if additional terms can improve cosine similarity
    best_terms_sorted = term_df['term'].tolist()
    best_term = best_terms_sorted[0]
    terms_to_check = list(set(best_terms_sorted[1:]))
    best_cos_sim = term_df['score'].iloc[0]
    terms_to_include[img_name] = [best_term]

    # Perform a loop to find if additional terms can improve the cosine similarity
    n_iteration = 5
    for iteration in range(n_iteration):
        data_list = []
        for term_to_test in terms_to_check:
            new_term = f"{best_term} {term_to_test}"
            score = clip_manager.get_image_caption_score(new_term, img_feat_dic[img_name])
            data_list.append({
                'term': new_term, 'candidate': term_to_test, 'score': score
            })
        combined_df = pd.DataFrame(data_list).sort_values('score', ascending=False)
        if combined_df['score'].iloc[0] > best_cos_sim + 0.01:
            best_cos_sim = combined_df['score'].iloc[0]
            terms_to_include[img_name].append(combined_df['candidate'].iloc[0])
            terms_to_check = combined_df['candidate'].tolist()[1:]
            best_term += f" {combined_df['candidate'].iloc[0]}"
        else:
            break


# #### Generate captions

# In[ ]:


num_captions = 50

# Set LM params
model_params = {'temperature': 0.9, 'max_length': 40, 'do_sample': True}

# Create dictionaries to store the outputs
prompt_dic = {}
sorted_caption_map = {}
caption_score_map = {}

for img_name in img_dic:
    # Create the prompt for the language model
    prompt_dic[img_name] = pg.create_improved_lm_prompt(
        img_type_dic[img_name], num_people_dic[img_name], terms_to_include[img_name]
    )

    # Generate the caption using the language model
    caption_texts = flan_manager.generate_response(num_captions * [prompt_dic[img_name]], model_params)

    # Zero-shot VLM: rank captions.
    caption_feats = clip_manager.get_text_feats(caption_texts)
    sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_feats, img_feat_dic[img_name])
    sorted_caption_map[img_name] = sorted_captions
    caption_score_map[img_name] = dict(zip(sorted_captions, caption_scores))


# ### Save the outputs

# In[ ]:


data_list = []
for img_name in img_dic:
    generated_caption = sorted_caption_map[img_name][0]
    data_list.append({
        'image_name': img_name,
        'image_path': img_paths[img_name],
        'generated_caption': generated_caption,
        'cosine_similarity': caption_score_map[img_name][generated_caption]
    })
pd.DataFrame(data_list).to_csv(f'{approach}_outputs.csv', index=False)


# In[ ]:




