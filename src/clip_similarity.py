from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F

# Load CLIP
model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print("CLIP loaded successfully")

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Two sample images from SAME item_id
img1_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

img2_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_2_side.jpg"

# Open images
image1 = Image.open(img1_path).convert("RGB")
image2 = Image.open(img2_path).convert("RGB")

# Preprocess
inputs1 = processor(images=image1, return_tensors="pt")
inputs2 = processor(images=image2, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    emb1 = model.get_image_features(**inputs1)
    emb2 = model.get_image_features(**inputs2)

# Extract tensor if wrapped
if hasattr(emb1, "pooler_output"):
    emb1 = emb1.pooler_output

if hasattr(emb2, "pooler_output"):
    emb2 = emb2.pooler_output

# Normalize
emb1 = emb1 / emb1.norm(p=2, dim=-1, keepdim=True)
emb2 = emb2 / emb2.norm(p=2, dim=-1, keepdim=True)

# Cosine similarity
similarity = F.cosine_similarity(emb1, emb2)

print("\nCosine Similarity:")
print(similarity.item())