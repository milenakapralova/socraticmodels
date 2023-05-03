# Package loading
import os
import requests
import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from profanity_filter import ProfanityFilter
import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ImageManager:
    def __init__(self):
        self.download_data()

    @staticmethod
    def download_data():
        # Download test image
        fname = 'demo_img.png'
        if not os.path.exists(fname):
            img_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg"
            with open(fname, 'wb') as f:
                f.write(requests.get(img_url).content)

    @staticmethod
    def load_image(image_path):
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


class VocabManager:
    def __init__(self):
        self.download_data()
        self.place_list = self.load_places()
        s = time.time()
        self.object_list = self.load_objects(remove_profanity=True)
        print(f'new object took: {time.time()-s}')
        s = time.time()
        self.object_list2 = self.original_load_objects()
        print(f'old object took: {time.time()-s}')
        print(f'equality: {set(self.object_list) == set(self.object_list2)}')


    @staticmethod
    def download_data():
        # Download scene categories from Places365.
        if not os.path.exists('categories_places365.txt'):
            url = "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"
            response = requests.get(url)
            with open("categories_places365.txt", "wb") as f:
                f.write(response.content)
        # Download object categories from Tencent ML Images.
        if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
            url = "https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt"
            response = requests.get(url)
            with open("dictionary_and_semantic_hierarchy.txt", "wb") as f:
                f.write(response.content)

    @staticmethod
    def load_places():
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        place_texts = []
        for place in place_categories[:, 0]:
            place = place.split('/')[2:]
            if len(place) > 1:
                place = place[1] + ' ' + place[0]
            else:
                place = place[0]
            place = place.replace('_', ' ')
            place_texts.append(place)
        return place_texts

    def load_objects(self, remove_profanity=False):
        with open('dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        object_texts = []
        pf = ProfanityFilter()
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            if remove_profanity:
                safe_list = ''
                for variant in object_text.split(','):
                    text = variant.strip()
                    if pf.is_clean(text):
                        safe_list += f'{text}, '

                safe_list = safe_list[:-2]
                if len(safe_list) > 0:
                    object_texts.append(safe_list)
            else:
                object_texts.append(object_text)
        return [o for o in list(set(object_texts)) if o not in self.place_list]


class ClipManager:
    def __init__(self, device):
        self.device = device
        self.feat_dim_map = {
            'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512,
            'ViT-B/16': 512,'ViT-L/14': 768
        }
        self.version = "ViT-L/14"
        self.feat_dim = self.feat_dim_map[self.version]
        self.model, self.preprocess = clip.load(self.version)
        self.model.to(self.device)
        self.model.eval()

    def get_text_feats(self, in_text, batch_size=64):
        text_tokens = clip.tokenize(in_text).to(self.device)
        text_id = 0
        text_feats = np.zeros((len(in_text), self.feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id + batch_size]
            with torch.no_grad():
                batch_feats = self.model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id:text_id + batch_size, :] = batch_feats
            text_id += batch_size
        return text_feats

    def get_img_feats(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        img_in = self.preprocess(img_pil)[None, ...]
        with torch.no_grad():
            img_feats = self.model.encode_image(img_in.to(self.device)).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return img_feats

    @staticmethod
    def get_nn_text(raw_texts, text_feats, img_feats):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores = np.sort(scores).squeeze()[::-1]
        return high_to_low_texts, high_to_low_scores


class FlanT5Manager:
    def __init__(self, version="google/flan-t5-xl"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(version)
        self.tokenizer = AutoTokenizer.from_pretrained(version)

    def generate_response(self, prompt, model_params=None):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **model_params)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def num_params(model):
    return np.sum([int(np.prod(p.shape)) for p in model.parameters()])


def print_clip_info(model):
    print("Model parameters (total):", num_params(model))
    print("Model parameters (image encoder):", num_params(model.visual))
    print("Model parameters (text encoder):", num_params(model.token_embedding) + num_params(model.transformer))
    print("Input image resolution:", model.visual.input_resolution)
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)


def main(img_path='demo_img.png', verbose=True):

    # Set the device to use
    if getattr(torch, 'has_mps', False):
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'gpu'
    else:
        device = 'cpu'

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
    ppl_texts = ['no people', 'people']
    ppl_feats = clip_manager.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
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
    place_topk = 3
    sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_feats, img_feats)

    # Zero-shot VLM: classify objects.
    obj_topk = 10
    sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_feats, img_feats)
    object_list = ''
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
    object_list = object_list[:-2]

    # Zero-shot LM: generate captions.
    num_captions = 10
    prompt = f'''I am an intelligent image captioning bot.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    A creative short caption I can generate to describe this image is:'''


    caption_texts = [
        flan_manager.generate_response(prompt, model_params={'temperature': 0.9, 'do_sample': True})
        for _ in range(num_captions)
    ]

    # Zero-shot VLM: rank captions.
    caption_feats = clip_manager.get_text_feats(caption_texts)
    sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_feats, img_feats)
    print(f'{sorted_captions[0]}\n')

    if verbose:
        print(f'VLM: This image is a:')
        for img_type, score in zip(sorted_img_types, img_type_scores):
            print(f'{score:.4f} {img_type}')

        print(f'\nVLM: There:')
        for ppl_text, score in zip(sorted_ppl_texts, ppl_scores):
            print(f'{score:.4f} {ppl_text}')

        print(f'\nVLM: I think this photo was taken at a:')
        for place, score in zip(sorted_places[:place_topk], places_scores[:place_topk]):
            print(f'{score:.4f} {place}')

        print(f'\nVLM: I think there might be a:')
        for obj_text, score in zip(sorted_obj_texts[:obj_topk], obj_scores[:obj_topk]):
            print(f'{score:.4f} {obj_text}')

        print(f'\nLM generated captions ranked by VLM scores:')
        for caption, score in zip(sorted_captions, caption_scores):
            print(f'{score:.4f} {caption}')

if __name__ == '__main__':
    main()