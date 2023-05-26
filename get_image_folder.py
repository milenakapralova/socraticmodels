'''
This file creates a folder with the 50 images used in the validation split.

'''

# Loading the reuqired packages
import os
import pandas as pd
import matplotlib.pyplot as plt




def load_caption(caption):
    """
    Loads a caption csv file from the csv name passed.
    """
    try:
        caption_df = pd.read_csv(caption)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{caption} not found! Please run the coco_captioning_baseline.py and coco_captioning_improved.py "
            "to obtain the generated captions before proceeding with the evaluation."
        )
    return caption_df


def load_all_captions():
    """
    Load all the captions in the '../data/outputs/captions/' directory.

    :return: Dictionary mapping caption csv file names to the loaded dataframe.
    """
    caption_dir = 'data/outputs/captions/'
    return {c.split('.')[0]: load_caption(caption_dir + c) for c in os.listdir(caption_dir)}

# Load the generated captions
caption_dic = load_all_captions()

# Extract the list of images
img_list = list(caption_dic.values())[0]['image_name'].tolist()




# Path to the directory containing the images
image_dir = 'data/coco/val2017/'

# Paths to the CSV files containing image captions
csv_file1 = 'data/outputs/captions/gpt_caption.csv'
csv_file2 = 'data/outputs/captions/baseline_caption.csv'
csv_file3 = 'data/outputs/captions/improved_caption.csv'

# Read the CSV files into pandas DataFrames
data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)
data3 = pd.read_csv(csv_file3)



captions1 = data1[data1['image_name'] == '000000100624.jpg']['generated_caption']

print('t')
# Iterate over each image file in the directory
for image_name in img_list:
    # Find the corresponding captions for the image in each DataFrame
    captions1 = data1[data1['image_name'] == image_name]['generated_caption']
    captions2 = data2[data2['image_name'] == image_name]['generated_caption']
    captions3 = data3[data3['image_name'] == image_name]['generated_caption']

    print('t')
    # Combine all the captions into a single list
    captions = list(captions1) + list(captions2) + list(captions3)

    # Plot the image and display each caption below it
    image_path = os.path.join(image_dir, image_name)
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')

    for i, caption in enumerate(captions):
        plt.title(f'Caption {i+1}: {caption}')
        plt.show()
