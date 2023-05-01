from transformers import CLIPModel, CLIPProcessor
import torch

# Load CLIP model and processor: this is already done in the notebook
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare caption for processing
caption = "A cat sleeping on a couch"
inputs = processor(caption, return_tensors="pt", padding=True)

# Generate embedding for caption
with torch.no_grad():
    embedding = model.get_text_features(**inputs)[0]