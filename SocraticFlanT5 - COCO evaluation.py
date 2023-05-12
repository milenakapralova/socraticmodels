'''
The following script:
- Downloads the COCO images and annotations
- Passes the COCO images through the Socratic model pipeline and gets the resulting captions for the COCO images
- Evaluates the resulting captions against ground truth COCO annotations

Please make sure that you have the files place_feats.npy and object_feats.npy in your directory. If that is not the case, please run the generate_features.py file.
'''

# Package loading
from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager
from eval import SocraticEvalCap
from utils import get_device
import os
import re
import json
import numpy as np
import pickle
import time


# def main():
# Step 1: Downloading the COCO images and annotations
imgs_folder = 'imgs/val2017/'
annotation_file = 'annotations/annotations/captions_val2017.json'

# Step 2: Passing the COCO images through the Socratic model pipeline
## Set the device to use
device = get_device()

## Instantiate the clip manager
clip_manager = ClipManager(device)

## Instantiate the image manager
image_manager = ImageManager()

## Instantiate the vocab manager
vocab_manager = VocabManager()

## Calculate the place features
if not os.path.exists('cache/place_feats.npy'):

    # Calculate the place features
    place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])
    np.save('cache/place_feats.npy', place_feats)
else:
    place_feats = np.load('cache/place_feats.npy')

## Calculate the object features
if not os.path.exists('cache/object_feats.npy'):
    # Calculate the object features
    object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])
    np.save('cache/object_feats.npy', object_feats)
else:
    object_feats = np.load('cache/object_feats.npy')

## Defining parameters regarding the template prompt
### Zero-shot VLM: classify image type.
img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_feats = clip_manager.get_text_feats([f'This is a {t}.' for t in img_types])

### Zero-shot VLM: classify number of people.
ppl_texts = ['no people', 'people']
ppl_feats = clip_manager.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])

### Zero-shot VLM: classify places.
place_topk = 3

### Zero-shot VLM: classify objects.
obj_topk = 10

### Zero-shot LM: generate captions.
num_captions = 1

## Generating captions for images
if not os.path.exists('cache/res.pickle'):
    ## Instantiate the Flan T5 manager
    flan_manager = FlanT5Manager(version="google/flan-t5-xl", use_api=False)

    res = {}
    embed_imgs = {}
    embed_capt_res = {}

    # N = len(os.listdir(imgs_folder))
    N = 3

    for ix, file_name in enumerate(os.listdir(imgs_folder)[:N]):
        start_time = time.time()
        if file_name.endswith(".jpg"):  # consider only image files
            # Getting image id
            ## Image_id
            file_name_strip = file_name.strip('.jpg')
            match = re.search('^0+', file_name_strip)
            sequence = match.group(0)
            image_id = int(file_name_strip[len(sequence):])

            img_path = os.path.join(imgs_folder, file_name)
            img = image_manager.load_image(img_path)
            img_feats = clip_manager.get_img_feats(img)
            embed_imgs[image_id] = img_feats.flatten()

            # Zero-shot VLM: classify image type.
            sorted_img_types, img_type_scores = clip_manager.get_nn_text(img_types, img_types_feats, img_feats)
            img_type = sorted_img_types[0]

            # Zero-shot VLM: classify number of people.
            sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_feats, img_feats)
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'people':
                ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
                ppl_feats = clip_manager.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
                sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_feats, img_feats)
                ppl_result = sorted_ppl_texts[0]
            else:
                ppl_result = f'are {ppl_result}'

            # Zero-shot VLM: classify places.
            sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_feats, img_feats)

            # Zero-shot VLM: classify objects.
            sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_feats, img_feats)
            object_list = ''
            for i in range(obj_topk):
                object_list += f'{sorted_obj_texts[i]}, '
            object_list = object_list[:-2]

            # Zero-shot LM: generate captions.
            prompt = f'''I am an intelligent image captioning bot.
            This image is a {img_type}. There {ppl_result}.
            I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
            I think there might be a {object_list} in this {img_type}.
            A creative short caption I can generate to describe this image is:'''

            # Generate multiple captions
            model_params = {'temperature': 0.9, 'max_length': 40, 'do_sample': True}
            caption_texts = flan_manager.generate_response(num_captions * [prompt], model_params)

            # Zero-shot VLM: rank captions.
            caption_feats = clip_manager.get_text_feats(caption_texts)
            sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_feats, img_feats)
            best_caption = sorted_captions[0]

            res[image_id] = [{
                'image_id': image_id,
                'id': image_id,
                'caption': best_caption
            }]
            embed_capt_res[image_id] = clip_manager.get_text_feats([best_caption]).flatten()

            print(f'time taken {time.time()-start_time}')

    with open('cache/res.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('cache/embed_imgs.pickle', 'wb') as handle:
        pickle.dump(embed_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('cache/embed_capt_res.pickle', 'wb') as handle:
        pickle.dump(embed_capt_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('cache/res.pickle', 'rb') as handle:
        res = pickle.load(handle)

# Step 3: Evaluating the resulting captions against ground truth COCO annotations
## Load the ground truth annotations
with open(annotation_file, 'r') as f:
    lines = json.load(f)['annotations']
gts = {}
for item in lines:
    if item['image_id'] not in gts:
        gts[item['image_id']] = []
    gts[item['image_id']].append({'image_id': item['image_id'], 'caption': item['caption']})

# If the embeddings for the gt captions are not yet computed, compute then
if not os.path.exists('cache/embed_capt_gt.pickle'):
    embed_capt_gt = {}
    for img_id, list_of_capt_dict in gts.items():
        list_of_captions = [capt_dict['caption'] for capt_dict in list_of_capt_dict]

        # Dims of img_feats_gt: 5 x 768
        img_feats_gt = clip_manager.get_text_feats(list_of_captions)

        embed_capt_gt[img_id] = img_feats_gt

    with open('cache/embed_capt_gt.pickle', 'wb') as handle:
        pickle.dump(embed_capt_gt, handle, protocol=pickle.HIGHEST_PROTOCOL)

eval_cap = {}
evaluator = SocraticEvalCap(gts, res)

## Rule-based metrics
eval_rulebased = {}
evaluator.evaluate_rulebased()
for metric, score in evaluator.eval.items():
    print(f'{metric}: {score:.3f}')
    eval_rulebased[metric] = round(score, 5)
eval_cap['rulebased'] = eval_rulebased

## Metric based on cosine similarity/dot product between the captions and images
eval_cossim = {}
evaluator.evaluate_cossim()
for source_caption, sim in evaluator.sims_mean.items():
    # print(f'{source_caption}: avg = {sim[0]:.3f}, sd = {sim[1]:.3f}')
    # eval_cossim[source_caption] = [round(sim[0], 5), round(sim[1], 5)]
    print(f'{source_caption}: avg = {sim:.3f}')
    eval_cossim[source_caption] = [round(sim, 5)]

eval_cap['cossim'] = eval_cossim


## Save the evaluation scores
with open('eval_cap.pickle', 'wb') as handle:
    pickle.dump(eval_cap, handle, protocol=pickle.HIGHEST_PROTOCOL)



# if __name__ == '__main__':
#     main()