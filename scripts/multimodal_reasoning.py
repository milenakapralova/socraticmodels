from scripts.image_captioning import CocoManager, ClipManager, ImageManager, VocabManager, LmManager, GptManager, \
    LmPromptGenerator
# Local imports
from scripts.utils import print_time_dec, prepare_dir, set_all_seeds, get_device



class ImageCaptionerParent:
    """
    This is the parent class of the ImageCaptionerBaseline and ImageCaptionerImproved classes. It contains the
    functionality that is common to both child classes: the constructor of the class.
    """
    def __init__(self, random_seed=42, n_images=50, set_type='train'):
        """
        The constructor instantiates all the helper classes needed for the captioning. It sets the random seeds.
        Loads the vocabulary embeddings (these never change, so they are loaded from a cache). It loads a different set
        of images depending on the 'set_type' input. The embeddings of the images are derived. The cosine similarities
        between the text and image embeddings are determined. The remaining of the image captioning process is done
        by the children image captioning classes.

        Importantly, this method calculates all of the outputs that are independent of the hyperparameter choice for the
        image captioning process. This means this is only done once for the entire run during a hyperparameter tuning
        run.

        :param random_seed: The random seed that will be used in order to ensure deterministic results.
        :param n_images: The number of images that will be captioned.
        :param set_type: The data set type (train/valid/test).
        """

        """
        1. Set up
        """
        # Store set type
        self.set_type = set_type

        # Set the seeds
        set_all_seeds(random_seed)

        # ## Step 1: Downloading the MS COCO images and annotations
        self.coco_manager = CocoManager()

        # ### Set the device, instantiate managers and calculate the variables that are image independent.

        # Set the device to use
        device = get_device()

        # Instantiate the clip manager
        self.clip_manager = ClipManager(device)

        # Instantiate the image manager
        self.image_manager = ImageManager()

        # Instantiate the vocab manager
        self.vocab_manager = VocabManager()

        # Instantiate the language model manager
        self.lm_manager = LmManager()

        # Instantiate the GPT manager
        self.gpt_manager = GptManager()

        # Instantiate the prompt generator
        self.prompt_generator = LmPromptGenerator()

        # Set up the prompt generator map
        self.pg_map = {
            'original': self.prompt_generator.create_socratic_original_prompt,
            'creative': self.prompt_generator.create_improved_lm_creative,
            'gpt': self.prompt_generator.create_gpt_prompt_likely,
        }

        """
        2. Text embeddings
        """

        # Calculate the place features
        self.place_emb = CacheManager.get_place_emb(self.clip_manager, self.vocab_manager)

        # Calculate the object features
        self.object_emb = CacheManager.get_object_emb(self.clip_manager, self.vocab_manager)

        # Calculate the features of the number of people
        self.ppl_texts = None
        self.ppl_emb = None
        self.ppl_texts_bool = None
        self.ppl_emb_bool = None
        self.ppl_texts_mult = None
        self.ppl_emb_mult = None
        self.get_nb_of_people_emb()

        # Calculate the features for the image types
        self.img_types = ['photo', 'cartoon', 'sketch', 'painting']
        self.img_types_emb = self.clip_manager.get_text_emb([f'This is a {t}.' for t in self.img_types])

        # Create a dictionary that maps the objects to the cosine sim.
        self.object_embeddings = dict(zip(self.vocab_manager.object_list, self.object_emb))

        """
        3. Load images and compute image embedding
        """
        self.science_qa_dataset = [
            sample for sample in load_dataset('derek-thomas/ScienceQA', split='validation')
            if sample['image'] is not None
        ]



    def get_nb_of_people_emb(self):
        """
        Gets the embeddings for the number of people.

        Method to be overriden in the child class.

        :return:
        """
        pass

    def determine_nb_of_people(self):
        """
        Determines the number of people in the image.

        Method to be overriden in the child class.

        :return:
        """
        pass

    def show_demo_image(self, img_name):
        """
        Creates a visualisation of the image using matplotlib.

        :param img_name: Input image to show
        :return:
        """
        # Show the image
        plt.imshow(self.img_dic[self.image_manager.image_folder + img_name])
        plt.show()