'''
The following script:
- Downloads the COCO images and annotations
- Passes the COCO images through the Socratic model pipeline
- Gets the resulting captions for the COCO images
- Evaluates the resulting captions afainst ground truth COCO annotations
'''

# Package loading
from image_captioning import COCOManager
# ClipManager, ImageManager, VocabManager, FlanT5Manager

# Step 1: Downloading the COCO images and annotations
coco_manager = COCOManager()




