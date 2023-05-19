'''
SocraticFlanT5 - Caption Generation (improved) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the improved Socratic model pipeline:
an improved baseline model where the template prompt filled by CLIP is processed before passing to FLAN-T5-xl

'''

# Package loading
from transformers import set_seed
import os
import numpy as np
import pandas as pd

# Local imports
import sys

sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass
from scripts.image_captioning import (
    ClipManager, ImageManager, VocabManager, FlanT5Manager, CocoManager, LmPromptGenerator
)
from scripts.image_captioning import CacheManager as cm
from scripts.utils import get_device, prepare_dir, set_all_seeds, get_file_name_extension, print_time_dec


@print_time_dec
def main(
        num_images=50, num_captions=30, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True, cos_sim_thres=0.7,
        num_objects=5, num_places=2, caption_strategy='baseline', random_seed=42
):
    """
    1. Set up
    """
    # Set the seeds
    set_all_seeds(random_seed)

    # ## Step 1: Downloading the MS COCO images and annotations
    coco_manager = CocoManager()

    # ## Step 2: Generating the captions via the Socratic pipeline

    # ### Set the device and instantiate managers

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

    # Instantiate the prompt generator
    prompt_generator = LmPromptGenerator()

    """
    2. Text embeddings
    """

    # Calculate the place features
    place_emb = cm.get_place_emb(clip_manager, vocab_manager)

    # Calculate the object features
    object_emb = cm.get_object_emb(clip_manager, vocab_manager)

    """
    3. Load images and compute image embedding
    """

    # Randomly select images from the COCO dataset
    img_files = coco_manager.get_random_image_paths(num_images=num_images)

    # Create dictionaries to store the images features
    img_dic = {}
    img_feat_dic = {}

    for img_file in img_files:
        # Load the image
        img_dic[img_file] = image_manager.load_image(coco_manager.image_dir + img_file)
        # Generate the CLIP image embedding
        img_feat_dic[img_file] = clip_manager.get_img_emb(img_dic[img_file]).flatten()

    """
    4. Zero-shot VLM (CLIP): We zero-shot prompt CLIP to produce various inferences of an image, such as image type or 
    the number of people in the image.
    """

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

    # Create a dictionary to store the location
    location_dic = {}
    for img_name, img_feat in img_feat_dic.items():
        sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_emb, img_feat)
        location_dic[img_name] = sorted_places

    # Classify image object

    # Create a dictionary to store the similarity of each object with the images
    object_score_map = {}
    sorted_obj_dic = {}
    for img_name, img_feat in img_feat_dic.items():
        sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_emb, img_feat)
        object_score_map[img_name] = dict(zip(sorted_obj_texts, obj_scores))
        sorted_obj_dic[img_name] = sorted_obj_texts

    """
    5. Finding both relevant and different objects using cosine similarity
    """

    # Create a dictionary that maps the objects to the cosine sim.
    object_embeddings = dict(zip(vocab_manager.object_list, object_emb))

    # Create a dictionary to store the best object matches
    best_matches = {}

    for img_name, sorted_obj_texts in sorted_obj_dic.items():

        # Create a list that contains the objects ordered by cosine sim.
        embeddings_sorted = [object_embeddings[w] for w in sorted_obj_texts]

        # Create a list to store the best matches
        best_matches[img_name] = [sorted_obj_texts[0]]

        # Create an array to store the embeddings of the best matches
        unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

        # Loop through the 100 best objects by cosine similarity
        for i in range(1, 100):
            # Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
            max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
            # If object i is different enough to the current best matches, add it to the best matches
            if max_cos_sim < cos_sim_thres:
                unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
                best_matches[img_name].append(sorted_obj_texts[i])

    """
    6. Zero-shot LM (Flan-T5): We zero-shot prompt Flan-T5 to produce captions and use CLIP to rank the captions
    generated
    """
    # Set up the prompt generator map
    pg_map = {
        'baseline': prompt_generator.create_baseline_lm_prompt2,
    }

    # Set LM params
    model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

    # Create dictionaries to store the outputs
    prompt_dic = {}
    sorted_caption_map = {}
    caption_score_map = {}

    for img_name in img_dic:
        prompt_dic[img_name] = pg_map[caption_strategy](
            img_type_dic[img_name], num_people_dic[img_name], location_dic[img_name][:num_places],
            object_list=best_matches[img_name][:num_objects]
        )

        # Generate the caption using the language model
        caption_texts = flan_manager.generate_response(num_captions * [prompt_dic[img_name]], model_params)

        # Zero-shot VLM: rank captions.
        caption_emb = clip_manager.get_text_emb(caption_texts)
        sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_emb, img_feat_dic[img_name])
        sorted_caption_map[img_name] = sorted_captions
        caption_score_map[img_name] = dict(zip(sorted_captions, caption_scores))

    data_list = []
    for img_name in img_dic:
        generated_caption = sorted_caption_map[img_name][0]
        data_list.append({
            'image_name': img_name,
            'generated_caption': generated_caption,
            'cosine_similarity': caption_score_map[img_name][generated_caption]
        })
    file_name_extension = get_file_name_extension(
        lm_temperature, cos_sim_thres, num_objects, num_places, caption_strategy
    )
    file_path = f'../data/outputs/captions/improved_caption{file_name_extension}.csv'
    prepare_dir(file_path)
    pd.DataFrame(data_list).to_csv(file_path, index=False)


if __name__ == '__main__':

    template_params = dict(
        num_images=50, num_captions=30, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True, cos_sim_thres=0.7,
        num_objects=5, num_places=2, caption_strategy='baseline', random_seed=42
    )

    # Run with the base parameters
    main(**template_params)

    # Temperature search
    for t in (0.85, 0.95):
        temp_params = template_params.copy()
        temp_params['lm_temperature'] = t
        main(**temp_params)

    # Cosine similarity threshold search
    for c in (0.6, 0.8):
        temp_params = template_params.copy()
        temp_params['cos_sim_thres'] = c
        main(**temp_params)

    # Number of generated objects search
    for n in (4, 6, 7):
        temp_params = template_params.copy()
        temp_params['num_objects'] = n
        main(**temp_params)

    # Number of places search
    for n in (1, 3):
        temp_params = template_params.copy()
        temp_params['num_places'] = n
        main(**temp_params)

