
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch

from image_captioning import ImageManager
from utils import get_device

# Instantiate the image manager
image_manager = ImageManager()

# Load image.
img_path = 'monkey_with_gun.jpg'
img = image_manager.load_image(img_path)

# Set the device to use
device = get_device()

# Example 1

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: Describe the image: Answer:"
inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, torch.float16)

outputs = model(**inputs)

# Example 2

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = processor(img, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)