# Imports
import os
from typing import List, Union
import requests
import clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv
import json
from scripts.utils import print_time_dec


class ImageManager:
    def __init__(self, img_dir='data/sample_images', download_samples=False):
        """
        images_to_download: image_path to download_url map
        """
        os.makedirs(img_dir, exist_ok=True)
        self.images_to_download = {
            f'{img_dir}/demo_img.png': 'https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg',
            f'{img_dir}/monkey_with_gun.jpg': 'https://drive.google.com/uc?export=download&id=1iG0TJTZ0yRJEC8dA-WwS7X-GhuX8sfy8',
            f'{img_dir}/astronaut_with_beer.jpg': 'https://drive.google.com/uc?export=download&id=1p5RwifMFtl1CLlUXaIR_y60_laDTbNMi',
            f'{img_dir}/fruit_bowl.jpg': 'https://drive.google.com/uc?export=download&id=1gRYMoTfCwuV4tNy14Qf2Q_hebx05GNd9'
        }
        
        if download_samples:
            self.download_data()

    def download_data(self):
        """
        Downloads the images of self.images_to_download if the file does not exist.
        :return:
        """
        # Download images
        for img_path, img_url in self.images_to_download.items():
            if not os.path.exists(img_path):
                self.download_image_from_url(img_path, img_url)

    @staticmethod
    def download_image_from_url(img_path: str, img_url: str):
        """
        Downloads an image from an url.

        :param img_path: Output path.
        :param img_url: Download url.
        :return:
        """
        with open(img_path, 'wb') as f:
            f.write(requests.get(img_url).content)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """

        :param image_path:
        :return:
        """
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


class VocabManager:
    def __init__(self, data_dir='data'):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.download_data()
        self.place_list = self.load_places()
        self.object_list = self.load_objects()

    def download_data(self):
        # Download scene categories from Places365.
        if not os.path.exists(f'{self.data_dir}/categories_places365.txt'):
            url = "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"
            response = requests.get(url)
            with open(f'{self.data_dir}/categories_places365.txt', 'wb') as f:
                f.write(response.content)
        # Download object categories from Tencent ML Images.
        if not os.path.exists(f'{self.data_dir}/dictionary_and_semantic_hierarchy.txt'):
            url = (
                "https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierar"
                "chy.txt"
            )
            response = requests.get(url)
            with open(f'{self.data_dir}/dictionary_and_semantic_hierarchy.txt', 'wb') as f:
                f.write(response.content)

    def load_places(self) -> List[str]:
        place_categories = np.loadtxt(f'{self.data_dir}/categories_places365.txt', dtype=str)
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

    def load_objects(self) -> List[str]:
        with open(f'{self.data_dir}/dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        object_texts = []
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            object_texts.append(object_text)
        return [o for o in list(set(object_texts)) if o not in self.place_list]


class ClipManager:
    def __init__(self, device: str, version: str = "ViT-L/14"):
        """
        :param device: The device to use ('gpu', 'mps', 'cpu').
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
    
    def get_img_info(self, img, place_feats, obj_feats, vocab_manager, place_topk=3, obj_topk=10):
        # get image features
        img_feats = self.get_img_feats(img)
        # classify image type
        img_types = ['photo', 'cartoon', 'sketch', 'painting']
        img_types_feats = self.get_text_feats([f'This is a {t}.' for t in img_types])
        sorted_img_types, img_type_scores = self.get_nn_text(img_types, img_types_feats, img_feats)
        img_type = sorted_img_types[0]
        print(f'This is a {img_type}.')

        # classify number of people
        ppl_texts = [
            'are no people', 'is one person', 'are two people', 'are three people', 'are several people', 'are many people'
        ]
        ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
        num_people = sorted_ppl_texts[0]
        print(f'There {num_people} in this photo.')

        # classify places
        sorted_places, places_scores = self.get_nn_text(vocab_manager.place_list, place_feats, img_feats)
        location = sorted_places[0]
        print(f'It was taken in {location}.')

        # classify objects
        sorted_obj_texts, obj_scores = self.get_nn_text(vocab_manager.object_list, obj_feats, img_feats)
        object_list = ''
        for i in range(obj_topk):
            object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]
        print(f'Top 10 objects in the image: \n{sorted_obj_texts[:10]}')
        
        return img_type, num_people, location, sorted_obj_texts, object_list, obj_scores
    
    def rank_gen_outputs(self, img, output_texts, k=5):
        img_feats = self.get_img_feats(img)
        output_feats = self.get_text_feats(output_texts)
        sorted_outputs, output_scores = self.get_nn_text(output_texts, output_feats, img_feats)
        output_score_map = dict(zip(sorted_outputs, output_scores))
        for i, output in enumerate(sorted_outputs[:k]):
            print(f'{i + 1}. {output} ({output_score_map[output]:.2f})')
    
class LMManager:
    def __init__(self, version, use_api=False, device='cpu'):
        self.model = None
        self.tokenizer = None
        self.api_url = None
        self.headers = None
        self.use_api = use_api
        self.device = device
        if use_api:
            load_dotenv()
            try:
                hf_api = os.getenv('HUGGINGFACE_API')
            except ValueError:
                    "You need to store your huggingface api key in your environment under "
                    "'HUGGINGFACE_API' if you want to use the API. Otherwise, set 'use_api' to False."
                
            self.api_url = f"https://api-inference.huggingface.co/models/{version}"
            self.headers = {"Authorization": f"Bearer {hf_api}"}
        else:
            # Instantiate the model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(version)
            self.model.to(self.device)
            # Instantiate the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(version)

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
        :param prompt: Prompt(s) as list or str
        :param model_params: Model parameters
        :return: str if 1 else list
        """
        if model_params is None:
            model_params = {}
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **model_params)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(decoded) == 1:
            return decoded[0]
        return decoded

    def generate_response_api(
            self, prompt: Union[List[str], str], model_params: Union[dict, None] = None
    ) -> Union[List[str], str]:
        """
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
        # json.loads(response.content.decode("utf-8"))

class CacheManager:
    @staticmethod
    def get_place_emb(clip_manager, vocab_manager):
        place_emb_path = '../data/cache/place_emb.npy'
        if not os.path.exists(place_emb_path):
            # Ensure the directory exists
            os.makedirs(place_emb_path, exist_ok=True)
            # Calculate the place features
            place_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.place_list])
            np.save(place_emb_path, place_emb)
        else:
            # Load cache
            place_emb = np.load(place_emb_path)
        return place_emb

    @staticmethod
    def get_object_emb(clip_manager, vocab_manager):
        object_emb_path = '../data/cache/object_emb.npy'
        if not os.path.exists(object_emb_path):
            # Ensure the directory exists
            os.makedirs(object_emb_path, exist_ok=True)
            # Calculate the place features
            object_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.object_list])
            np.save(object_emb_path, object_emb)
        else:
            # Load cache
            object_emb = np.load(object_emb_path)
        return object_emb

    @staticmethod
    def get_img_emb(clip_manager, vocab_manager):
        object_emb_path = '../data/cache/object_emb.npy'
        if not os.path.exists(object_emb_path):
            # Ensure the directory exists
            os.makedirs(object_emb_path, exist_ok=True)
            # Calculate the place features
            object_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.object_list])
            np.save(object_emb_path, object_emb)
        else:
            # Load cache
            object_emb = np.load(object_emb_path)
        return object_emb
    
class PromptGenerator:
    @staticmethod
    def gen_baseline_prompt(img_type, ppl_result, sorted_places, object_list_str):
        return f'''I am an intelligent image captioning bot.
        This image is a {img_type}. There {ppl_result}.
        I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
        I think there might be a {object_list_str} in this {img_type}.
        A creative short caption I can generate to describe this image is:'''

    @staticmethod
    def gen_improved_prompt(img_type, ppl_result, sorted_places, filtered_objs):
        return f'''I am a poetic writer that creates image captions.
        This image is a {img_type}. There {ppl_result}.
        This photo may have been taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
        There might be a {', '.join(filtered_objs)} in this {img_type}.
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
    
def filter_objs(sorted_obj_texts, obj_scores, clip_manager, obj_topk=10, sim_threshold=0.7):
    '''
    Filter unique objects in image using cosine similarity between object embeddings.
    Input:
        sorted_obj_texts: list of objects in image sorted by scores
        obj_scores: clip scores for objects
        clip_manager: clip manager
        obj_topk: number of objects to keep
        sim_threshold: cosine similarity threshold for similarity
    Output:
        filtered_objs: list of filtered objects
    '''
    sorted_obj_indices = np.argsort(obj_scores)[::-1]

    unique_obj_indices = []
    # Rest of the code...

    # Extract individual words
    individual_obj_texts = []
    for obj in sorted_obj_texts:
        individual_obj_texts.extend([word.strip() for word in obj.split(',')])
    # individual_obj_texts = extract_individual_words(sorted_obj_texts)
    individual_obj_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in individual_obj_texts])

    for obj_idx in sorted_obj_indices:
        if len(unique_obj_indices) == obj_topk:
            break

        is_similar = False
        for unique_obj_idx in unique_obj_indices:
            similarity = individual_obj_feats[obj_idx].dot(individual_obj_feats[unique_obj_idx])
            if similarity >= sim_threshold:
                is_similar = True
                break

        if not is_similar:
            unique_obj_indices.append(obj_idx)

    unique_objects = [individual_obj_texts[i] for i in unique_obj_indices]
    object_list = ', '.join(unique_objects)
    # print(f'objects in image: {object_list}')

    return unique_objects

def filter_objs_alt(obj_list, sorted_obj_texts, obj_feats, img_feats, clip_manager, obj_topk=10, sim_threshold=0.7):
	'''
	Filter unique objects in image using cosine similarity between object embeddings.
	Input:
		obj_list: list of objects in vocabulary
		sorted_obj_texts: list of objects in image sorted by scores
		obj_feats: object embeddings
		img_feats: image embeddings
		clip_manager: clip manager
		obj_topk: number of objects to keep
		sim_threshold: cosine similarity threshold for similarity
	Output:
		filtered_objs: list of filtered objects
	'''
	# Create a dictionary that maps the objects to the cosine sim.
	obj_embeddings = dict(zip(obj_list, obj_feats))

	# Create a list that contains the objects ordered by cosine sim.
	embeddings_sorted = [obj_embeddings[w] for w in sorted_obj_texts]

	# Create a list to store the best matches
	best_matches = [sorted_obj_texts[0]]

	# Create an array to store the embeddings of the best matches
	unique_embeddings = embeddings_sorted[0].reshape(-1, 1)

	# Loop through the 100 best objects by cosine similarity
	for i in range(1, 100):
		# Obtain the maximum cosine similarity when comparing object i to the embeddings of the current best matches
		max_cos_sim = (unique_embeddings.T @ embeddings_sorted[i]).max()
		# If object i is different enough to the current best matches, add it to the best matches
		if max_cos_sim < sim_threshold:
			unique_embeddings = np.concatenate([unique_embeddings, embeddings_sorted[i].reshape(-1, 1)], 1)
			best_matches.append(sorted_obj_texts[i])

	# Looping through the best matches, consider each terms separately by splitting the commas and spaces.
	# init clip manager
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# clip_manager = ClipManager(device=device)
	data_list = []
	for terms in best_matches:
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
	filtered_objs = [best_term]

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
			filtered_objs.append(combined_df['candidate'].iloc[0])
			terms_to_check = combined_df['candidate'].tolist()[1:]
			best_term += f" {combined_df['candidate'].iloc[0]}"
		else:
			break
	# print(f'filtered terms: {filtered_objs}')

	return filtered_objs 