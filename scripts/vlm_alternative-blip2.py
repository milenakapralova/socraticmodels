import os
import sys
sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass
from image_captioning import ImageManager, Blip2Manager
from utils import get_device

# Set the device to use
device = get_device()

# Instantiate the BLIP2 manager
blip2_manager = Blip2Manager(device)

# Instantiate the image manager
image_manager = ImageManager()

# Load the image
img_folder = '../data/images/example_images/'
img_file = 'astronaut_with_beer.jpg'
img_path = img_folder + img_file
image = image_manager.load_image(img_path)

# Set the model parameters
model_params = {
    'max_length': 40,
    'no_repeat_ngram_size': 2,
    'repetition_penalty': 1.5,
}

# Example 1: Caption without a prompt
caption = blip2_manager.generate_response(image, model_params=model_params)
print(f'BLIP2 caption without prompt: "{caption}"')

# Example 2: Asking questions to BLIP2
question1 = "Question: where is the picture taken? Answer:"
response1 = blip2_manager.generate_response(image, prompt=question1, model_params=model_params)
print(f'Prompt input: "{question1}" - BLIP2 Output: "{response1}"')

question2 = "Question: what are the different objects in the image? Answer:"
response2 = blip2_manager.generate_response(image, prompt=question2, model_params=model_params)
print(f'Prompt input: "{question2}" - BLIP2 Output: "{response2}"')

question3 = "Question: Who is in the image? Answer:"
response3 = blip2_manager.generate_response(image, prompt=question3, model_params=model_params)
print(f'Prompt input: "{question3}" - BLIP2 Output: "{response3}"')
