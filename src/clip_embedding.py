from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print("CLIP loaded successfully")

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Sample image
img_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

# Open image
image = Image.open(img_path).convert("RGB")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Convert to tensor if needed
if hasattr(image_features, "pooler_output"):
    image_features = image_features.pooler_output

# Normalize embedding
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

print("\nEmbedding Shape:")
print(image_features.shape)

print("\nFirst 10 values:")
print(image_features[0][:10])