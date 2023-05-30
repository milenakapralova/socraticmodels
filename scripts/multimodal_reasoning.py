# Package loading
import sys
import os
import argparse
sys.path.append('..')

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
from scripts.image_captioning import ImageCaptionerParent


class MmReasoner(ImageCaptionerParent):
    def __init__(self):
        super().__init__(set_type='mm_reasoning')

    def create_cot_prompt_from_sample(self, sample):
        # compose prompt
        # Generate the CLIP image embedding
        img_emb = self.clip_manager.get_img_emb(sample['image']).flatten()

        # Obtain a list of objects ordered by cosine similarity with the image
        sorted_obj_texts, obj_scores = self.clip_manager.get_nn_text(
            self.vocab_manager.object_list, self.object_emb, img_emb
        )
        # Obtain a list of places ordered by cosine similarity with the image
        sorted_places, places_scores = self.clip_manager.get_nn_text(
            self.vocab_manager.place_list, self.place_emb, img_emb
        )
        # Create the prompt
        return self.prompt_generator.create_cot_prompt(sample, sorted_places, sorted_obj_texts)

    def create_vqa_prompt_from_sample(self, sample):
        # compose prompt
        # Generate the CLIP image embedding
        img_emb = self.clip_manager.get_img_emb(sample['image']).flatten()

        # Obtain a list of objects ordered by cosine similarity with the image
        sorted_obj_texts, obj_scores = self.clip_manager.get_nn_text(
            self.vocab_manager.object_list, self.object_emb, img_emb
        )
        # Obtain a list of places ordered by cosine similarity with the image
        sorted_places, places_scores = self.clip_manager.get_nn_text(
            self.vocab_manager.place_list, self.place_emb, img_emb
        )
        # Create the prompt
        return self.prompt_generator.create_vqa_prompt(sample, sorted_places, sorted_obj_texts)
