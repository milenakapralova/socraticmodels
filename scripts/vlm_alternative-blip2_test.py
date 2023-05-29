"""
This file was used to obtain the benchmarks for the BLIP2 model.

It uses the ImageCaptionerBaseline to ensure that the selected test images are the same for all the
different model/hyperparameter runs.
"""
# Import packages.
import os
import sys
import pandas as pd
sys.path.append('..')
# Depending on the platform/IDE used, the home directory might be the socraticmodels or the
# socraticmodels/scripts directory. The following ensures that the current directory is the scripts folder.
try:
    os.chdir('scripts')
except FileNotFoundError:
    pass
from scripts.image_captioning import Blip2Manager
from scripts.coco_caption_base_hp_tune import ImageCaptionerBaseline


from scripts.utils import get_device

# Set the device to use
device = get_device()

# Instantiate the BLIP2 manager
blip2_manager = Blip2Manager(device)

# Instantiate the baseline image captioner to easily load the test images and calculate the cosine similarities
image_captioner = ImageCaptionerBaseline(n_images=50, set_type='test')

# Set the model parameters
model_params = {
    'max_length': 40,
    'no_repeat_ngram_size': 2,
    'repetition_penalty': 1.5,
}

# Create an empty list to store the data
data_list = []
# Loop through the test images
for img_path, img in image_captioner.img_dic.items():
    # Generate a caption for the image with BLIP2
    caption = blip2_manager.generate_response(img, model_params=model_params)
    # Add the data sample
    data_list.append({
        'image_name': img_path.split('/')[-1],
        'generated_caption': caption,
        'cosine_similarity': image_captioner.clip_manager.get_image_caption_score(
            caption, image_captioner.img_feat_dic[img_path]
        )
    })
# Save the caption dataframe
file_path = f'../data/outputs/captions/blip2_caption_test.csv'
pd.DataFrame(data_list).to_csv(file_path, index=False)

