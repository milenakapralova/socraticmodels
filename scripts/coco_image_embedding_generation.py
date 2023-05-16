#!/usr/bin/env python
# coding: utf-8

# # SocraticFlanT5 - Image Embedding Generation | DL2 Project, May 2023
# ---
# 
# This notebook computes the embeddings for images of MS COCO and saves them in a `.pickle` file.

# ## Set-up
# If you haven't done so already, please activate the corresponding environment by running in the terminal: `conda env create -f environment.yml`. Then type `conda activate socratic`.

# ### Loading the required packages

# In[1]:


# Package loading
# import matplotlib.pyplot as plt
# import pandas as pd
# from transformers import set_seed
import os
import numpy as np
import re
import pickle
# import time
import random
# import pandas as pd

# Local imports
from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, COCOManager
from utils import get_device, prepare_dir


# ## Step 1: Downloading the MS COCO images and annotations

# In[3]:


imgs_folder = '../data/images/val2017/'

coco_manager = COCOManager()
coco_manager.download_data()


# ## Step 2: Generating the captions via the Socratic pipeline
# 

# ### Set the device and instantiate managers

# In[4]:


# Set the device to use
device = get_device()

# Instantiate the clip manager
clip_manager = ClipManager(device)

# Instantiate the image manager
image_manager = ImageManager()

# Instantiate the vocab manager
vocab_manager = VocabManager()


# ### Compute place and object features

# In[5]:

# Calculate the place features
file_path = '../data/cache/place_emb.npy'
if not os.path.exists(file_path):
    prepare_dir(file_path)
    # Calculate the place features
    place_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.place_list])
    np.save(file_path, place_emb)
else:
    place_emb = np.load(file_path)


# Calculate the object features
file_path = '../data/cache/object_emb.npy'
if not os.path.exists(file_path):
    prepare_dir(file_path)
    # Calculate the object features
    object_emb = clip_manager.get_text_emb([f'Photo of a {o}.' for o in vocab_manager.object_list])
    np.save(file_path, object_emb)
else:
    object_emb = np.load(file_path)


# ### Load images and compute image embedding

# In[ ]:


embed_imgs = {}

for ix, file_name in enumerate(os.listdir(imgs_folder)): 
        # Getting image id
        file_name_strip = file_name.strip('.jpg')
        match = re.search('^0+', file_name_strip)
        sequence = match.group(0)
        image_id = int(file_name_strip[len(sequence):])

        img_path = os.path.join(imgs_folder, file_name)
        img = image_manager.load_image(img_path)
        img_emb = clip_manager.get_img_emb(img)
        img_emb = img_emb.flatten()
        embed_imgs[image_id] = img_emb


# ### Save the outputs

# In[ ]:

file_path = '../data/cache/object_emb.npy'
prepare_dir(file_path)
with open(file_path, 'wb') as handle:
        pickle.dump(embed_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




