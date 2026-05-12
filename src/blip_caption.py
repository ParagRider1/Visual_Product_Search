from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch

# Load BLIP model
model_name = "Salesforce/blip-image-captioning-base"

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

print("BLIP loaded successfully")

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Sample image
img_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

# Open image
image = Image.open(img_path).convert("RGB")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs)

# Decode caption
caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

print("\nGenerated Caption:")
print(caption)