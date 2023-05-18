"""
This file contains the "failed" improved pipeline. This pipeline allows to create a prompt that maximises the CLIP
cosine similarity with the image. However, the caption generated did not perform well when tested on the COCO dataset.
Therefore, this pipeline idea was abandoned.
"""


'''
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the improved Socratic model pipeline:
an improved baseline model where the template prompt filled by CLIP is processed before passing to FLAN-T5-xl

'''
# Set-up
# If you haven't done so already, please activate the corresponding environment by running in the terminal: `conda env create -f environment.yml`. Then type `conda activate socratic`.

# Package loading
import os
import numpy as np
import pandas as pd
try:
    os.chdir('scripts')
except:
    pass

# Local imports
import sys
sys.path.append('..')
from scripts.image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, CocoManager
from scripts.image_captioning import LmPromptGenerator as pg
from scripts.image_captioning import CacheManager as cm
from scripts.utils import get_device, prepare_dir, set_all_seeds, get_uuid_for_imgs


# Set the seeds
set_all_seeds(42)


# Step 1: Downloading the MS COCO images and annotations
coco_manager = CocoManager()

# Step 2: Generating the captions via the Socratic pipeline

# Set the device and instantiate managers

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

# Calculate the place features
place_emb = cm.get_place_emb(clip_manager, vocab_manager)

# Calculate the object features
object_emb = cm.get_object_emb(clip_manager, vocab_manager)


# Load images and compute image embedding

# Randomly select images from the COCO dataset
N = 5
img_files = coco_manager.get_random_image_paths(num_images=5)

# Create dictionaries to store the images features
img_dic = {}
img_feat_dic = {}

for img_file in img_files:
    # Load the image
    img_dic[img_file] = image_manager.load_image(coco_manager.image_dir + img_file)
    # Generate the CLIP image embedding
    img_feat_dic[img_file] = clip_manager.get_img_emb(img_dic[img_file]).flatten()


# Zero-shot VLM (CLIP)
# We zero-shot prompt CLIP to produce various inferences of an iage, such as image type or the number of people in an image:

# Classify image type
img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_emb = clip_manager.get_text_emb([f'This is a {t}.' for t in img_types])

# Create a dictionary to store the image types
img_type_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_img_types, img_type_scores = clip_manager.get_nn_text(img_types, img_types_emb, img_feat)
    img_type_dic[img_name] = sorted_img_types[0]


# Classify number of people
ppl_texts = [
    'are no people', 'is one person', 'are two people', 'are three people', 'are several people', 'are many people'
]
ppl_emb = clip_manager.get_text_emb([f'There {p} in this photo.' for p in ppl_texts])

# Create a dictionary to store the number of people
num_people_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_emb, img_feat)
    num_people_dic[img_name] = sorted_ppl_texts[0]


# Classify image place
place_topk = 3

# Create a dictionary to store the number of people
location_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_emb, img_feat)
    location_dic[img_name] = sorted_places[0]


# Classify image object
obj_topk = 10

# Create a dictionary to store the similarity of each object with the images
object_score_map = {}
sorted_obj_dic = {}
for img_name, img_feat in img_feat_dic.items():
    sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_emb, img_feat)
    object_score_map[img_name] = dict(zip(sorted_obj_texts, obj_scores))
    sorted_obj_dic[img_name] = sorted_obj_texts


# Finding both relevant and different objects using cosine similarity

# Create a dictionary that maps the objects to the cosine sim.
object_embeddings = dict(zip(vocab_manager.object_list, object_emb))

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


# Generate captions
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
    caption_emb = clip_manager.get_text_emb(caption_texts)
    sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_emb, img_feat_dic[img_name])
    sorted_caption_map[img_name] = sorted_captions
    caption_score_map[img_name] = dict(zip(sorted_captions, caption_scores))


# Store the captions
data_list = []
for img_name in img_dic:
    generated_caption = sorted_caption_map[img_name][0]
    data_list.append({
        'image_name': img_name,
        'generated_caption': generated_caption,
        'cosine_similarity': caption_score_map[img_name][generated_caption]
    })
file_path = f'../data/outputs/captions/improved_caption.csv'
prepare_dir(file_path)
pd.DataFrame(data_list).to_csv(file_path, index=False)
