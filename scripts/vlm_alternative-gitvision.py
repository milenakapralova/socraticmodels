
from transformers import AutoProcessor, GitVisionModel
from image_captioning import ImageManager

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = GitVisionModel.from_pretrained("microsoft/git-base")

# Instantiate the image manager
image_manager = ImageManager()

# Load image.
img_folder = '../data/images/example_images/'
img_file = 'monkey_with_gun.jpg'
img_path = img_folder + img_file
img = image_manager.load_image(img_path)

inputs = processor(images=img, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
