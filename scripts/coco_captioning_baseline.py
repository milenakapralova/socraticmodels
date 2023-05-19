'''
SocraticFlanT5 - Caption Generation (baseline) | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the baseline Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is replaced by FLAN-T5-xl.

Set-up
If you haven't done so already, please activate the corresponding environment by running in the terminal:
`conda env create -f environment.yml`. Then type `conda activate socratic`.
'''

# Package loading
import pandas as pd
import sys
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except:
    pass

# Local imports
from scripts.image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, CocoManager
from scripts.image_captioning import LmPromptGenerator as pg
from scripts.image_captioning import CacheManager as cm
from scripts.utils import get_device, prepare_dir, set_all_seeds, print_time_dec


@print_time_dec
def main(num_images=50, num_captions=30, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True, random_seed=42):

    """
    1. Set up
    """

    # Set the seeds
    set_all_seeds(random_seed)

    # Step 1: Downloading the MS COCO images and annotations
    coco_manager = CocoManager()

    # Step 2: Generating the captions via the Socratic pipeline

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
    ppl_texts_bool = ['no people', 'people']
    ppl_emb_bool = clip_manager.get_text_emb([f'There are {p} in this photo.' for p in ppl_texts_bool])
    ppl_texts_mult = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
    ppl_emb_mult = clip_manager.get_text_emb([f'There {p} in this photo.' for p in ppl_texts_mult])

    # Create a dictionary to store the number of people
    num_people_dic = {}

    for img_name, img_feat in img_feat_dic.items():
        sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts_bool, ppl_emb_bool, img_feat)
        ppl_result = sorted_ppl_texts[0]
        if ppl_result == 'people':
            sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts_mult, ppl_emb_mult, img_feat)
            ppl_result = sorted_ppl_texts[0]
        else:
            ppl_result = f'are {ppl_result}'

        num_people_dic[img_name] = ppl_result

    # Classify image place

    # Create a dictionary to store the location
    location_dic = {}
    for img_name, img_feat in img_feat_dic.items():
        sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_emb, img_feat)
        location_dic[img_name] = sorted_places[0]

    # Classify image objects
    obj_topk = 10

    # Create a dictionary to store the similarity of each object with the images
    obj_list_dic = {}
    for img_name, img_feat in img_feat_dic.items():
        sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_emb, img_feat)
        object_list = ''
        for i in range(obj_topk):
            object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]
        obj_list_dic[img_name] = object_list

    """
    5. Zero-shot LM (Flan-T5): We zero-shot prompt Flan-T5 to produce captions and use CLIP to rank the captions
    generated
    """

    # Set LM params
    model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

    # Create dictionaries to store the outputs
    prompt_dic = {}
    sorted_caption_map = {}
    caption_score_map = {}

    for img_name in img_dic:
        # Create the prompt for the language model
        prompt_dic[img_name] = pg.create_baseline_lm_prompt(
            img_type_dic[img_name], num_people_dic[img_name], location_dic[img_name], obj_list_dic[img_name]
        )

        # Generate the caption using the language model
        caption_texts = flan_manager.generate_response(num_captions * [prompt_dic[img_name]], model_params)

        # Zero-shot VLM: rank captions.
        caption_emb = clip_manager.get_text_emb(caption_texts)
        sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_emb, img_feat_dic[img_name])
        sorted_caption_map[img_name] = sorted_captions
        caption_score_map[img_name] = dict(zip(sorted_captions, caption_scores))

    """
    6. Outputs.
    """

    # Store the captions
    data_list = []
    for img_name in img_dic:
        generated_caption = sorted_caption_map[img_name][0]
        data_list.append({
            'image_name': img_name,
            'generated_caption': generated_caption,
            'cosine_similarity': caption_score_map[img_name][generated_caption]
        })

    file_path = f'../data/outputs/captions/baseline_caption.csv'
    prepare_dir(file_path)
    pd.DataFrame(data_list).to_csv(file_path, index=False)


if __name__ == '__main__':
    main()
