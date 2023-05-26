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
import numpy as np
import sys
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except:
    pass

# Local imports
from scripts.image_captioning import ClipManager, ImageManager, VocabManager, LmManager, CocoManager, LmPromptGenerator, \
    ImageCaptionerParent
from scripts.image_captioning import CacheManager as cm
from scripts.utils import get_device, prepare_dir, set_all_seeds, print_time_dec, get_file_name_extension_baseline


class ImageCaptionerBaseline(ImageCaptionerParent):
    @print_time_dec
    def main(
            self, num_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True,
            num_objects=10, num_places=3
    ):

        # Set LM params
        model_params = {'temperature': lm_temperature, 'max_length': lm_max_length, 'do_sample': lm_do_sample}

        # Create dictionaries to store the outputs
        prompt_dic = {}
        sorted_caption_map = {}
        caption_score_map = {}

        for img_name in self.img_dic:

            prompt_dic[img_name] = self.prompt_generator.create_socratic_original_prompt(
                self.img_type_dic[img_name], self.num_people_dic[img_name], self.location_dic[img_name][:num_places],
                self.sorted_obj_dic[img_name][:num_objects]
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

        """
        6. Outputs.
        """

        # Store the captions
        data_list = []
        for img_name in self.img_dic:
            generated_caption = sorted_caption_map[img_name][0]
            data_list.append({
                'image_name': img_name,
                'generated_caption': generated_caption,
                'cosine_similarity': caption_score_map[img_name][generated_caption]
            })

        file_name_extension = get_file_name_extension_baseline(
            lm_temperature, num_objects, num_places
        )
        file_path = f'../data/outputs/captions/baseline_caption{file_name_extension}.csv'
        prepare_dir(file_path)
        pd.DataFrame(data_list).to_csv(file_path, index=False)

    def get_nb_of_people_emb(self):
        # Classify number of people
        self.ppl_texts_bool = ['no people', 'people']
        self.ppl_emb_bool = self.clip_manager.get_text_emb([
            f'There are {p} in this photo.' for p in self.ppl_texts_bool
        ])
        self.ppl_texts_mult = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
        self.ppl_emb_mult = self.clip_manager.get_text_emb([f'There {p} in this photo.' for p in self.ppl_texts_mult])

    def determine_nb_of_people(self):
        """
        Determines the number of people in the image.

        :return:
        """
        # Create a dictionary to store the number of people
        num_people_dic = {}
        for img_name, img_feat in self.img_feat_dic.items():
            sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(
                self.ppl_texts_bool, self.ppl_emb_bool, img_feat
            )
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'people':
                sorted_ppl_texts, ppl_scores = self.clip_manager.get_nn_text(
                    self.ppl_texts_mult, self.ppl_emb_mult, img_feat
                )
                ppl_result = sorted_ppl_texts[0]
            else:
                ppl_result = f'are {ppl_result}'

            self.num_people_dic[img_name] = ppl_result

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
            template_params_copy['num_objects'] = np.random.choice(range(5, 15))
            template_params_copy['num_places'] = np.random.choice(range(1, 6))
            self.main(**template_params_copy)



if __name__ == '__main__':

    image_captioner = ImageCaptionerBaseline(num_images=50, set_type='train')
    template_params = dict(
        num_images=50, num_captions=10, lm_temperature=0.9, lm_max_length=40, lm_do_sample=True, random_seed=42,
        num_objects=10, num_places=3
    )
    image_captioner.random_parameter_search(n_rounds=200, template_params=template_params)
