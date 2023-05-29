"""
SocraticGPT-3 - Caption Generation | DL2 Project, May 2023
This script downloads the images from the validation split of the MS COCO Dataset (2017 version)
and the corresponding ground-truth captions and generates captions based on the original Socratic model pipeline:
a Socratic model based on the work by Zeng et al. (2022) where GPT-3 is used as a LLM.

Set-up
If you haven't done so already, please activate the corresponding environment by running in the terminal:
`conda env create -f environment.yml`. Then type `conda activate socratic`.
"""

# Package loading
import sys
sys.path.append('..')
import os
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.coco_caption_base_hp_tune import ImageCaptionerBaseline


class ImageCaptionerGTP(ImageCaptionerBaseline):
    def generate_lm_response(self, prompt_list, model_params):
        return [
            self.gpt_manager.generate_response(
                prompt, temperature=model_params['temperature'], max_tokens=64, stop=None
            )
            for prompt in prompt_list
        ]

    def get_output_file_name(self, lm_temperature, n_objects, n_places, caption_strategy):
        extension = ''
        # The language model temperature
        extension += f'_temp_{lm_temperature}'.replace('.', '')
        # Number of objects
        extension += f'_nobj_{n_objects}'
        # Number of places
        extension += f'_npl_{n_places}'
        # Caption strategy
        extension += f'_strat_{caption_strategy}'
        # Train/test set
        extension += f'_{self.set_type}'
        return f'../data/outputs/captions/gpt_caption{extension}.csv'


if __name__ == '__main__':
    image_captioner = ImageCaptionerGTP(n_images=50, set_type='test')
    caption_dir = '../data/outputs/captions/'
    run_params = {
        'n_captions': 10,
        'lm_temperature': 0.9,
        'caption_strategy': 'gpt'
    }
    image_captioner.main(**run_params)
