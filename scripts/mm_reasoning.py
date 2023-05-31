# Package loading
import sys
import os
from datasets import load_dataset
sys.path.append('..')

# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass

# Local imports
import scripts.image_captioning as ic
from scripts.utils import set_all_seeds, get_device


class MmReasoner():
    def __init__(self, random_seed=42):
        # Set the seeds
        set_all_seeds(random_seed)

        # ### Set the device, instantiate managers and calculate the variables that are image independent.

        # Set the device to use
        device = get_device()
        
        # Load ScienceQA dataset
        self.sqa_dataset = load_dataset('derek-thomas/ScienceQA', split='validation')
        self.sqa_dataset = [sample for sample in self.sqa_dataset if sample['image'] is not None]
        
        # Instantiate the clip manager
        self.clip_manager = ic.ClipManager(device)

        # Instantiate the vocab manager
        self.vocab_manager = ic.VocabManager()

        # Instantiate the GPT manager
        self.gpt_manager = ic.GptManager()

        # Instantiate the prompt generator
        self.prompt_generator = ic.LmPromptGenerator()
        
        # Calculate the place features
        self.place_emb = ic.CacheManager.get_place_emb(self.clip_manager, self.vocab_manager)

        # Calculate the object features
        self.object_emb = ic.CacheManager.get_object_emb(self.clip_manager, self.vocab_manager)

    def create_cot_prompt(self, sample):
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

    def create_vqa_prompt(self, sample):
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
