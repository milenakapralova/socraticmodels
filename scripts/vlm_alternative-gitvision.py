import os
import sys

from scripts.utils import get_device

sys.path.append('..')
try:
    os.chdir('scripts')
except:
    pass
from scripts.image_captioning import ImageManager, GitVisionManager

# Set the device to use
device = get_device()

# Instantiate the GITVision manager
git_vision_manager = GitVisionManager(device)

# Instantiate the image manager
image_manager = ImageManager()

# Load image.
img_folder = '../data/images/example_images/'
img_file = 'monkey_with_gun.jpg'
img_path = img_folder + img_file
img = image_manager.load_image(img_path)

caption = git_vision_manager.generate_response(img)
print(f'GITVision caption: {caption}')
