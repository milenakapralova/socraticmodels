# Package loading
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import set_seed

from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, print_clip_info
from utils import get_device


# def main(img_path='demo_img.png', verbose=True):


img_path = '../monkey_with_gun.jpg'
verbose = True

# Set HuggingFace seed
set_seed(42)

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

# Print out clip model info
print_clip_info(clip_manager.model)

# Calculate the place features
place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])

# Calculate the object features
object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])

# Load image.
img = image_manager.load_image(img_path)
img_feats = clip_manager.get_img_feats(img)
plt.imshow(img)
plt.show()

# Zero-shot VLM: classify image type.
img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_feats = clip_manager.get_text_feats([f'This is a {t}.' for t in img_types])
sorted_img_types, img_type_scores = clip_manager.get_nn_text(img_types, img_types_feats, img_feats)
img_type = sorted_img_types[0]

# Zero-shot VLM: classify number of people.
ppl_texts = [
    'are no people', 'is one person', 'are two people', 'are three people', 'are several people', 'are many people'
]
ppl_feats = clip_manager.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_feats, img_feats)
ppl_result = sorted_ppl_texts[0]

# Zero-shot VLM: classify places.
place_topk = 3
sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_feats, img_feats)
place_score_map = dict(zip(sorted_places, places_scores))

# Zero-shot VLM: classify objects.
obj_topk = 10
sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_feats, img_feats)
object_score_map = dict(zip(sorted_obj_texts, obj_scores))
object_list = ''
for i in range(obj_topk):
    object_list += f'{sorted_obj_texts[i]}, '
object_list = object_list[:-2]

# Create a dictionary that maps the objects to the cosine sim.
object_embeddings = dict(zip(vocab_manager.object_list, object_feats))

# Create a list that contains the objects ordered by cosine sim.
top_100_embeddings = [object_embeddings[w] for w in sorted_obj_texts]

# Create a list to store the best matches
best_matches_from_top_100 = [sorted_obj_texts[0]]

# Create an array to store the embeddings of the best matches
unique_embeddings = top_100_embeddings[0].reshape(-1, 1)

# Loop through the 100 best objects by cosine similarity
for i in range(1, 100):
    # Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
    max_cos_sim = (unique_embeddings.T @ top_100_embeddings[i]).max()
    # If object i is different enough to the current best matches, add it to the best matches
    if max_cos_sim < 0.7:
        print(f'{sorted_obj_texts[i]}: {unique_embeddings.T @ top_100_embeddings[i]}')
        unique_embeddings = np.concatenate([unique_embeddings, top_100_embeddings[i].reshape(-1, 1)], 1)
        best_matches_from_top_100.append(sorted_obj_texts[i])

# Looping through the best matches, consider each terms separately by splitting the commas and spaces.
data_list = []
for terms in best_matches_from_top_100:
    for term_split in terms.split(', '):
        score = clip_manager.get_image_caption_score(term_split, img_feats)
        data_list.append({
            'term': term_split, 'score': score, 'context': terms
        })
        term_split_split = term_split.split(' ')
        if len(term_split_split) > 1:
            for term_split2 in term_split_split:
                score = clip_manager.get_image_caption_score(term_split2, img_feats)
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
terms_to_include = [best_term]

# Perform a loop to find if additional terms can improve the cosine similarity
n_iteration = 5
for iteration in range(n_iteration):
    data_list = []
    for term_to_test in terms_to_check:
        new_term = f"{best_term} {term_to_test}"
        score = clip_manager.get_image_caption_score(new_term, img_feats)
        data_list.append({
            'term': new_term, 'candidate': term_to_test, 'score': score
        })
    combined_df = pd.DataFrame(data_list).sort_values('score', ascending=False)
    if combined_df['score'].iloc[0] > best_cos_sim:
        best_cos_sim = combined_df['score'].iloc[0]
        terms_to_include.append(combined_df['candidate'].iloc[0])
        terms_to_check = combined_df['candidate'].tolist()[1:]
        best_term += f" {combined_df['candidate'].iloc[0]}"
    else:
        break



english_vocab = pd.read_csv('English-Morph.txt', sep=' ')

active_verbs = []
with open('English-Morph.txt', 'r') as file:
    lines = file.readlines()
    for l in lines:
        for w in l.split('\t'):
            if len(w) > 4 and w.endswith('ing'):
                active_verbs.append(w)

active_verbs = list(set(active_verbs))

active_verbs_fea = clip_manager.get_text_feats(active_verbs)
active_verbs_texts, active_verbs_scores = clip_manager.get_nn_text(active_verbs, active_verbs_fea, img_feats)

active_verb_map = dict(zip(active_verbs_texts, active_verbs_scores))

if len(terms_to_include) > 1:
    data_list = []
    for v in active_verbs_texts[:100]:

        test_term = f'{terms_to_include[0]} {v} {terms_to_include[1]}'

        score = clip_manager.get_image_caption_score(test_term, img_feats)

        data_list.append({'verb': v, 'new_term': test_term, 'score': score})

    verb_df = pd.DataFrame(data_list).sort_values('score', ascending=False)

