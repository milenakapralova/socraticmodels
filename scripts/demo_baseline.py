# Package loading
import matplotlib.pyplot as plt
import torch

from image_captioning import ClipManager, ImageManager, VocabManager, LmManager, print_clip_info
from utils import get_device


image_folder = '../data/coco/val2017/'
img_file = '000000244750.jpg'
img_path = image_folder + img_file
verbose = True

#
# def main(img_path='demo_img.png', verbose=True):
# Set the device to use
device = get_device()

# Instantiate the clip manager
clip_manager = ClipManager(device)

# Instantiate the image manager
image_manager = ImageManager()

# Instantiate the vocab manager
vocab_manager = VocabManager()

# Instantiate the Flan T5 manager
flan_manager = LmManager()

# Print out clip model info
print_clip_info(clip_manager.model)

# Calculate the place features
place_emb = clip_manager.get_text_emb([f'Photo of a {p}.' for p in vocab_manager.place_list])

# Calculate the object features
object_emb = clip_manager.get_text_emb([f'Photo of a {o}.' for o in vocab_manager.object_list])

# Load image.

img = image_manager.load_image(img_path)
img_emb = clip_manager.get_img_emb(img)
plt.imshow(img)
plt.show()

# Zero-shot VLM: classify image type.
img_types = ['photo', 'cartoon', 'sketch', 'painting']
img_types_emb = clip_manager.get_text_emb([f'This is a {t}.' for t in img_types])
sorted_img_types, img_type_scores = clip_manager.get_nn_text(img_types, img_types_emb, img_emb)
img_type = sorted_img_types[0]

# Zero-shot VLM: classify number of people.
ppl_texts = ['no people', 'people']
ppl_emb = clip_manager.get_text_emb([f'There are {p} in this photo.' for p in ppl_texts])
sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_emb, img_emb)
ppl_result = sorted_ppl_texts[0]
if ppl_result == 'people':
    ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
    ppl_emb = clip_manager.get_text_emb([f'There {p} in this photo.' for p in ppl_texts])
    sorted_ppl_texts, ppl_scores = clip_manager.get_nn_text(ppl_texts, ppl_emb, img_emb)
    ppl_result = sorted_ppl_texts[0]
else:
    ppl_result = f'are {ppl_result}'

# Zero-shot VLM: classify places.
place_topk = 3
sorted_places, places_scores = clip_manager.get_nn_text(vocab_manager.place_list, place_emb, img_emb)

# Zero-shot VLM: classify objects.
obj_topk = 10
sorted_obj_texts, obj_scores = clip_manager.get_nn_text(vocab_manager.object_list, object_emb, img_emb)
object_list = ''
for i in range(obj_topk):
    object_list += f'{sorted_obj_texts[i]}, '
object_list = object_list[:-2]

# Zero-shot LM: generate captions.
n_captions = 10
prompt = f'''I am an intelligent image captioning bot.
This image is a {img_type}. There {ppl_result}.
I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
I think there might be a {object_list} in this {img_type}.
A creative short caption I can generate to describe this image is:'''

# Generate multiple captions
model_params = {'temperature': 0.9, 'max_length': 40, 'do_sample': True}
caption_texts = flan_manager.generate_response(n_captions * [prompt], model_params)

# Zero-shot VLM: rank captions.
caption_emb = clip_manager.get_text_emb(caption_texts)
sorted_captions, caption_scores = clip_manager.get_nn_text(caption_texts, caption_emb, img_emb)
print(f'{sorted_captions[0]}\n')

if verbose:
    print(f'VLM: This image is a:')
    for img_type, score in zip(sorted_img_types, img_type_scores):
        print(f'{score:.4f} {img_type}')

    print(f'\nVLM: There:')
    for ppl_text, score in zip(sorted_ppl_texts, ppl_scores):
        print(f'{score:.4f} {ppl_text}')

    print(f'\nVLM: I think this photo was taken at a:')
    for place, score in zip(sorted_places[:place_topk], places_scores[:place_topk]):
        print(f'{score:.4f} {place}')

    print(f'\nVLM: I think there might be a:')
    for obj_text, score in zip(sorted_obj_texts[:obj_topk], obj_scores[:obj_topk]):
        print(f'{score:.4f} {obj_text}')

    print(f'\nLM generated captions ranked by VLM scores:')
    for caption, score in zip(sorted_captions, caption_scores):
        print(f'{score:.4f} {caption}')


# if __name__ == '__main__':
#     main()