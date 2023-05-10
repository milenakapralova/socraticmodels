'''
The following script:
- Downloads the COCO images and annotations
- Passes the COCO images through the Socratic model pipeline
- Gets the resulting captions for the COCO images
- Evaluates the resulting captions afainst ground truth COCO annotations
'''

# Package loading
from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, COCOManager
from utils import get_device
import os
import re


def main():
    # Step 1: Downloading the COCO images and annotations
    coco_manager = COCOManager()
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

    ## Instantiate the Flan T5 manager
    flan_manager = FlanT5Manager()

    ## Calculate the place features
    place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])

    ## Calculate the object features
    object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])

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
    num_captions = 10

    ## Generating captions for images
    res = []
    N = 5

    folder_path = 'imgs/val2017/'
    for ix, file_name in enumerate(os.listdir(folder_path)):
        if ix >= N:  # iterate only the first N images
            break
        if file_name.endswith(".jpg"):  # consider only image files
            img_path = os.path.join(folder_path, file_name)
            img = image_manager.load_image(img_path)
            img_feats = clip_manager.get_img_feats(img)

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

            # Getting image id's
            ## Image_id
            file_name = file_name.strip('.jpg')
            match = re.search('^0+', file_name)
            sequence = match.group(0)
            image_id = file_name[len(sequence):]

            ## Id
            id = ...

            current_caption = {'image_id': image_id,
                               'id': id,
                               'caption': best_caption
            }
            res.append(current_caption)

    return res










if __name__ == '__main__':
    main()