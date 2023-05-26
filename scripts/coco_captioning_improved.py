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
from scripts.image_captioning import ImageCaptionerParent
from scripts.utils import prepare_dir, get_file_name_extension_improved, print_time_dec


class ImageCaptionerImproved(ImageCaptionerParent):
    @print_time_dec
    def main(
            self, num_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True,
            cos_sim_thres=0.5, num_objects=5, num_places=2, caption_strategy='baseline'
    ):
        """
        5. Finding both relevant and different objects using cosine similarity
        """
        best_matches = self.find_best_object_matches(cos_sim_thres)

        """
        6. Zero-shot LM (Flan-T5): We zero-shot prompt Flan-T5 to produce captions and use CLIP to rank the captions
        generated
        """
        # Set up the prompt generator map
        pg_map = {
            'original': self.prompt_generator.create_socratic_original_prompt,
            'creative': self.prompt_generator.create_experimental_poetic_prompt,
        }

        # Set LM params
        model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

        # Create dictionaries to store the outputs
        prompt_dic = {}
        sorted_caption_map = {}
        caption_score_map = {}

        for img_name in self.img_dic:
            prompt_dic[img_name] = pg_map[caption_strategy](
                self.img_type_dic[img_name], self.num_people_dic[img_name], self.location_dic[img_name][:num_places],
                object_list=best_matches[img_name][:num_objects]
            )

            # Generate the caption using the language model
            caption_texts = self.flan_manager.generate_response(num_captions * [prompt_dic[img_name]], model_params)

            # Zero-shot VLM: rank captions.
            caption_emb = self.clip_manager.get_text_emb(caption_texts)
            sorted_captions, caption_scores = self.clip_manager.get_nn_text(
                caption_texts, caption_emb, self.img_feat_dic[img_name]
            )
            sorted_caption_map[img_name] = sorted_captions
            caption_score_map[img_name] = dict(zip(sorted_captions, caption_scores))

        data_list = []
        for img_name in self.img_dic:
            generated_caption = sorted_caption_map[img_name][0]
            data_list.append({
                'image_name': img_name,
                'generated_caption': generated_caption,
                'cosine_similarity': caption_score_map[img_name][generated_caption],
                'set_type': self.set_type
            })
        file_name_extension = get_file_name_extension_improved(
            lm_temperature, cos_sim_thres, num_objects, num_places, caption_strategy, self.set_type
        )
        file_path = f'../data/outputs/captions/improved_caption{file_name_extension}.csv'
        prepare_dir(file_path)
        pd.DataFrame(data_list).to_csv(file_path, index=False)

    def find_best_object_matches(self, cos_sim_thres):
        """
        This method is integral to the ImageCaptionerImproved. It filters the objects to only returned
        terms that do not have too high of cosine similarity with each other. It is controled by the cos_sim_thres
        parameter.

        :param cos_sim_thres:
        :return:
        """
        # Create a dictionary to store the best object matches
        best_matches = {}

        for img_name, sorted_obj_texts in self.sorted_obj_dic.items():

            # Create a list that contains the objects ordered by cosine sim.
            embeddings_sorted = [self.object_embeddings[w] for w in sorted_obj_texts]

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
        return best_matches

    def get_nb_of_people_emb(self):
        """
        Determines the number of people in the image.

        :return:
        """
        self.ppl_texts = [
            'are no people', 'is one person', 'are two people', 'are three people', 'are several people',
            'are many people'
        ]
        self.ppl_emb = self.clip_manager.get_text_emb([f'There {p} in this photo.' for p in self.ppl_texts])

    def random_parameter_search(self, n_rounds, template_params):
        """
        Runs a random parameter search.

        :param n_rounds:
        :param template_params:
        :return:
        """
        for _ in range(n_rounds):
            template_params_copy = template_params.copy()
            template_params_copy['lm_temperature'] = np.round(np.random.uniform(0.5, 1), 3)
            template_params_copy['cos_sim_thres'] = np.round(np.random.uniform(0.6, 1), 3)
            template_params_copy['num_objects'] = np.random.choice(range(5, 15))
            template_params_copy['num_places'] = np.random.choice(range(1, 6))
            template_params_copy['caption_strategy'] = np.random.choice(['original', 'creative'])
            self.main(**template_params_copy)

    def determine_nb_of_people(self):
        """
        Determines the number of people in the image.

        :return:
        """
        self.num_people_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(self.ppl_texts, self.ppl_emb, img_feat)
            self.num_people_dic[img_name] = sorted_ppl_texts[0]

if __name__ == '__main__':

    image_captioner = ImageCaptionerImproved(num_images=50, set_type='train')
    template_params = dict(
        num_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True, cos_sim_thres=0.7,
        num_objects=5, num_places=2, caption_strategy='baseline'
    )
    image_captioner.random_parameter_search(n_rounds=200, template_params=template_params)

