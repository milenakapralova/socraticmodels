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
from scripts.image_captioning import ClipManager, VocabManager, GptManager, LmManager, LmPromptGenerator, CacheManager
from scripts.utils import set_all_seeds, get_device


class MmReasoner:
    def __init__(self, random_seed=42):
        """
        The MmReasoner constructor executes the setup to perform multimodal reasoning.

        It sets the randomness seed to obtain a reproducible outcome. Instantiates the ClipManager, VocabManager,
        GptManager and LmPromptGenerator.

        :param random_seed:The random seed to use to obtain reproducible results.
        """

        # Set the seeds
        set_all_seeds(random_seed)

        # ### Set the device, instantiate managers and calculate the variables that are image independent.

        # Set the device to use
        device = get_device()
        
        # Load ScienceQA dataset
        self.sqa_dataset = load_dataset('derek-thomas/ScienceQA', split='validation')
        self.sqa_dataset = [sample for sample in self.sqa_dataset if sample['image'] is not None]
        
        # Instantiate the clip manager
        self.clip_manager = ClipManager(device)

        # Instantiate the vocab manager
        self.vocab_manager = VocabManager()

        # Instantiate the GPT manager
        self.gpt_manager = GptManager()

        self.lm_manager = LmManager()

        # Instantiate the prompt generator
        self.prompt_generator = LmPromptGenerator()
        
        # Calculate the place features
        self.place_emb = CacheManager.get_place_emb(self.clip_manager, self.vocab_manager)

        # Calculate the object features
        self.object_emb = CacheManager.get_object_emb(self.clip_manager, self.vocab_manager)

    def create_cot_prompt(self, sample):
        """
        The create_cot_prompt method creates a Chain of Thought (CoT) prompt.

        It receives a sample of the ScienceQA dataset, generates a CLIP embedding from the
        sample image. It compares the vocabulary cosine similarity to the image, obtaining the object and place terms
        that are most similar to the image. Finally, it uses the prompt generator to form a CoT prompt and returns it.

        :param sample: The ScienceQA sample that will be used to create the CoT prompt.
        :return: The CoT prompt as a string.
        """
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
        """
        The create_vqa_prompt method creates a Visual Question-Answering (VQA) prompt.

        It receives a sample of the ScienceQA dataset, generates a CLIP embedding from the
        sample image. It compares the vocabulary cosine similarity to the image, obtaining the object and place terms
        that are most similar to the image. Finally, it uses the prompt generator to form a VQA prompt and returns it.

        :param sample: The ScienceQA sample that will be used to create the VQA prompt.
        :return: The CoT prompt as a string.
        """
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
