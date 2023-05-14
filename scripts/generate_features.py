'''
This script computes place and object features and stores them in .npy files

'''
# Package loading
from image_captioning import ClipManager, VocabManager
from utils import get_device
import numpy as np
import os

def main():
    # Set the device to use
    device = get_device()

    # Instantiate the clip manager
    clip_manager = ClipManager(device)

    # Instantiate the vocab manager
    vocab_manager = VocabManager()

    if not os.path.exists('place_feats.npy'):
        # Calculate the place features
        place_feats = clip_manager.get_text_feats([f'Photo of a {p}.' for p in vocab_manager.place_list])
        np.save('place_feats.npy', place_feats)

    if not os.path.exists('object_feats.npy'):
        # Calculate the object features
        object_feats = clip_manager.get_text_feats([f'Photo of a {o}.' for o in vocab_manager.object_list])
        np.save('object_feats.npy', object_feats)


if __name__ == '__main__':
    main()