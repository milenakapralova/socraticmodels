'''
The following script:
- Downloads the COCO images and annotations
- Passes the COCO images through the Socratic model pipeline
- Gets the resulting captions for the COCO images
- Evaluates the resulting captions afainst ground truth COCO annotations
'''

# Package loading
from image_captioning import ClipManager, ImageManager, VocabManager, FlanT5Manager, COCOManager
from utils import get_device


def main():
    # Step 1: Downloading the COCO images and annotations
    coco_manager = COCOManager()
    imgs_folder = 'imgs/val2017/'
    annotation_file = 'annotations/annotations/captions_val2017.json'

    # Step 2: Passing the COCO images through the Socratic model pipeline
    ## Set the device to use
    device = get_device()

    ## Instantiate the clip manager
    clip_manager = ClipManager(device)

    ## Instantiate the image manager
    image_manager = ImageManager()

    ## Instantiate the vocab manager
    vocab_manager = VocabManager()

    ## Instantiate the Flan T5 manager
    flan_manager = FlanT5Manager()

    ## Calculate the place features
    place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])

    ## Calculate the object features
    object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])

    ## Generating captions for images
    res = []
    N = 5
    for _ in range(N):
        ...










if __name__ == '__main__':
    main()