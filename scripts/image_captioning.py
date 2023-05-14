# Package loading
import os
from typing import List, Union

import requests
import clip
import cv2
import numpy as np
from PIL import Image
from profanity_filter import ProfanityFilter
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration
import sys
sys.path.append('..')
from scripts.utils import print_time_dec, prepare_dir
import zipfile


class COCOManager:
    def __init__(self):
        """
        dataset: dataset to download
        """
        self.dataset_to_download = {
            '../data/coco/val2017': 'http://images.cocodataset.org/zips/val2017.zip',
            '../data/coco/annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        self.download_data()

    def download_unzip_delete(self, folder, url):
        """
        Checks if the COCO data is there, otherwise it downloads and unzips the data.

        :param folder:
        :param url:
        :return:
        """
        if not os.path.exists(folder):
            prepare_dir(folder)
            response = requests.get(url)
            parent = '/'.join(folder.split('/')[:-1])
            with open(parent + '/zip.zip', "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(parent + '/zip.zip', "r") as zip_ref:
                zip_ref.extractall(parent)
            os.remove(parent + '/zip.zip')

    def download_data(self):
        """
        Downloads the images and annotations of the COCO dataset of interest if the file does not exist.
        """
        for folder, url in self.dataset_to_download.items():
            self.download_unzip_delete(folder, url)


class ImageManager:
    def __init__(self):
        """
        images_to_download: image_path to download_url map
        """
        self.image_folder = '../data/images/example_images/'
        self.images_to_download = {
            'demo_img.png': 'https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg',
            'monkey_with_gun.jpg': 'https://drive.google.com/uc?export=download&id=1iG0TJTZ0yRJEC8dA-WwS7X-GhuX8sfy8',
            'astronaut_with_beer.jpg': 'https://drive.google.com/uc?export=download&id=1p5RwifMFtl1CLlUXaIR_y60_laDTbNMi',
            'fruit_bowl.jpg': 'https://drive.google.com/uc?export=download&id=1gRYMoTfCwuV4tNy14Qf2Q_hebx05GNd9',
            'cute_bear.jpg': 'https://drive.google.com/uc?export=download&id=1WvgweWH_vSODLv2EOoXqGaHDcUKPDHbh',
        }
        self.download_data()

    def download_data(self):
        """
        Downloads the images of self.images_to_download if the file does not exist.
        :return:
        """
        # Download images
        for img_path, img_url in self.images_to_download.items():
            if not os.path.exists(self.image_folder + img_path):
                self.download_image_from_url(img_path, img_url)

    def download_image_from_url(self, img_path: str, img_url: str):
        """
        Downloads an image from an url.

        :param img_path: Output path.
        :param img_url: Download url.
        :return:
        """
        file_path = self.image_folder + img_path
        prepare_dir(file_path)
        with open(self.image_folder + img_path, 'wb') as f:
            f.write(requests.get(img_url).content)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """

        :param image_path:
        :return:
        """
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


class VocabManager:
    def __init__(self):
        self.vocab_folder = '../data/vocabulary/'
        self.cache_folder = '../data/cache/'
        self.files_to_download = {
            'categories_places365.txt': "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_pl"
            "aces365.txt",
            'dictionary_and_semantic_hierarchy.txt': "https://raw.githubusercontent.com/Tencent/tencent-ml-images/maste"
            "r/data/dictionary_and_semantic_hierarchy.txt"
        }
        self.download_data()
        self.place_list = self.load_places()
        self.object_list = self.load_objects(remove_profanity=False)

    def download_data(self):
        """
        Download the vocabularies.

        :return:
        """
        # Download the vocabularies
        for file_name, url in self.files_to_download.items():
            file_path = self.vocab_folder + file_name
            if not os.path.exists(file_path):
                self.download_vocab_from_url(file_path, url)

    @staticmethod
    def download_vocab_from_url(file_path, url):
        """
        Downloads a file for a given url and stores it in the file_path.

        :param file_path: Output file
        :param url: Download url
        :return:
        """
        prepare_dir(file_path)
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    @print_time_dec
    def load_places(self) -> List[str]:
        """
        Load the places.

        This function comes from the original Socratic Models repository. A cache was added to speed up execution.

        :return:
        """
        file_path = self.vocab_folder + 'categories_places365.txt'
        cache_path = self.cache_folder + 'place_texts.txt'
        # Ensure the cache folder exists
        prepare_dir(cache_path)
        if not os.path.exists(cache_path):
            # Load the raw places file
            place_categories = np.loadtxt(file_path, dtype=str)
            place_texts = []
            for place in place_categories[:, 0]:
                place = place.split('/')[2:]
                if len(place) > 1:
                    place = place[1] + ' ' + place[0]
                else:
                    place = place[0]
                place = place.replace('_', ' ')
                place_texts.append(place)
            # Cache the file for the next run
            with open(cache_path, 'w') as f:
                for place in place_texts:
                    f.write(f"{place}\n")
        else:
            # Read the cache file
            with open(cache_path) as f:
                place_texts = f.read().splitlines()
        return place_texts

    @print_time_dec
    def load_objects(self, remove_profanity: bool = False) -> List[str]:
        """
        Load the objects.

        This function comes from the original Socratic Models repository. A cache was added to speed up execution.

        :return:
        """
        file_path = self.vocab_folder + 'dictionary_and_semantic_hierarchy.txt'
        cache_path = self.cache_folder + 'object_texts.txt'
        # Ensure the cache folder exists
        prepare_dir(cache_path)
        if not os.path.exists(cache_path):
            # Load the raw object file
            with open(file_path) as fid:
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
            # Cache the file for the next run
            with open(cache_path, 'w') as f:
                for obj in object_texts:
                    f.write(f"{obj}\n")
        else:
            # Read the cache file
            with open(cache_path) as f:
                object_texts = f.read().splitlines()
        return [o for o in list(set(object_texts)) if o not in self.place_list]


class ClipManager:
    def __init__(self, device: str, version: str = "ViT-L/14"):
        """
        The ClipManager handles all the methods relating to the CLIP model.

        :param device: The device to use ('cuda', 'mps', 'cpu').
        :param version: The CLIP model version.
        """
        self.device = device
        self.feat_dim_map = {
            'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512,
            'ViT-B/16': 512,'ViT-L/14': 768
        }
        self.version = version
        self.feat_dim = self.feat_dim_map[version]
        self.model, self.preprocess = clip.load(version)
        self.model.to(self.device)
        self.model.eval()

    def get_text_feats(self, in_text: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Creates a numpy array of text features with the columns containing the features and the rows containing the
        representations for each of the strings in the input in_text list.

        :param in_text: List of prompts
        :param batch_size: The batch size
        :return: Array with n_features columns and len(in_text) rows
        """
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

    def get_image_caption_score(self, caption, img_feats):
        text_feats = self.get_text_feats([caption])
        return float(text_feats @ img_feats.T)


class FlanT5Manager:
    def __init__(self, version="google/flan-t5-xl", use_api=False):
        """
        The FlanT5Manager handles all the method related to the Flan T5 model.

        :param version:
        :param use_api:
        """
        self.model = None
        self.tokenizer = None
        self.api_url = None
        self.headers = None
        self.use_api = use_api
        if use_api:
            if 'HUGGINGFACE_API' in os.environ:
                hf_api = os.environ['HUGGINGFACE_API']
            else:
                raise ValueError(
                    "You need to store your huggingface api key in your environment under "
                    "'HUGGINGFACE_API' if you want to use the API. Otherwise, set 'use_api' to False."
                )
            self.api_url = f"https://api-inference.huggingface.co/models/{version}"
            self.headers = {"Authorization": f"Bearer {hf_api}"}
        else:
            # Instantiate the model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(version)
            # Instantiate the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(version)

    @print_time_dec
    def generate_response(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        if self.use_api:
            if isinstance(prompt, str):
                return self.generate_response_api(prompt, model_params)
            else:
                return [self.generate_response_api(p, model_params) for p in prompt]
        else:
            return self.generate_response_local(prompt, model_params)

    def generate_response_local(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
        Generates a response using a local model. Accepts a single prompt or a list of prompts.

        :param prompt: Prompt(s) as list or str
        :param model_params: Model parameters
        :return: str if 1 else list
        """
        if model_params is None:
            model_params = {}
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **model_params)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(decoded) == 1:
            return decoded[0]
        return decoded

    def generate_response_api(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
        Generate a response through the API. Accepts a single prompt or a list of prompts.

        :param prompt: Prompt(s) as list or str
        :param model_params: Model parameters
        :return: str if 1 else list
        """
        if model_params is None:
            model_params = {}
        outputs = self.query({
            "inputs": prompt,
            "parameters": model_params,
            "options": {"use_cache": False, "wait_for_model": True}
        })
        decoded = [output['generated_text'] for output in outputs][0]
        return decoded

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()


class Blip2Manager:
    def __init__(self, device, version="Salesforce/blip2-opt-2.7b"):
        self.processor = Blip2Processor.from_pretrained(version)
        self.model = Blip2ForConditionalGeneration.from_pretrained(version, torch_dtype=torch.float16)
        self.device = device

    def generate_response(self, image, prompt=None, model_params=None):
        """

        :param image: Input image.
        :param prompt: The prompt to pass to BLIP.
        :param model_params:
        :return:
        """
        if model_params is None:
            model_params = {}
        if prompt is None:
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float16)
        self.model.to(self.device)
        out = self.model.generate(**inputs, **model_params)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class LmPromptGenerator:
    @staticmethod
    def create_baseline_lm_prompt(img_type, ppl_result, sorted_places, object_list):
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
        I think there might be a {object_list} in this {img_type}.
        A creative short caption I can generate to describe this image is:'''

    @staticmethod
    def create_improved_lm_prompt(img_type, ppl_result, terms_to_include):
        return f'''Create a creative beautiful caption from this context:
        "This image is a {img_type}. There {ppl_result}.
        The context is: {', '.join(terms_to_include)}.
        A creative short caption I can generate to describe this image is:'''

def num_params(model):
    """
    Calculates the number of parameters in the model.

    :param model:
    :return: Int
    """
    return np.sum([int(np.prod(p.shape)) for p in model.parameters()])


def print_clip_info(model):
    print("Model parameters (total):", num_params(model))
    print("Model parameters (image encoder):", num_params(model.visual))
    print("Model parameters (text encoder):", num_params(model.token_embedding) + num_params(model.transformer))
    print("Input image resolution:", model.visual.input_resolution)
    print("Context length:", model.context_length)
    print("Vocab size:", model.vocab_size)
