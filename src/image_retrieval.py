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

# Query image
query_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

# Small gallery
gallery_paths = [
    DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_2_side.jpg",

    DATASET_ROOT / "Img" / "img" / "WOMEN" / "Skirts" / "id_00000003" / "02_1_front.jpg",

    DATASET_ROOT / "Img" / "img" / "MEN" / "Denim" / "id_00000080" / "01_1_front.jpg",
]

# Load query image
query_image = Image.open(query_path).convert("RGB")

query_inputs = processor(images=query_image, return_tensors="pt")

# Query embedding
with torch.no_grad():
    query_emb = model.get_image_features(**query_inputs)

if hasattr(query_emb, "pooler_output"):
    query_emb = query_emb.pooler_output

query_emb = query_emb / query_emb.norm(p=2, dim=-1, keepdim=True)

# Store similarities
results = []

# Compare against gallery
for gallery_path in gallery_paths:

    gallery_image = Image.open(gallery_path).convert("RGB")

    gallery_inputs = processor(images=gallery_image, return_tensors="pt")

    with torch.no_grad():
        gallery_emb = model.get_image_features(**gallery_inputs)

    if hasattr(gallery_emb, "pooler_output"):
        gallery_emb = gallery_emb.pooler_output

    gallery_emb = gallery_emb / gallery_emb.norm(p=2, dim=-1, keepdim=True)

    similarity = F.cosine_similarity(query_emb, gallery_emb)

    results.append({
        "image": gallery_path.name,
        "similarity": similarity.item()
    })

# Sort descending
results = sorted(results, key=lambda x: x["similarity"], reverse=True)

# Print results
print("\nTop Retrieval Results")
print("-" * 40)

for rank, result in enumerate(results, start=1):
    print(f"{rank}. {result['image']} --> {result['similarity']:.4f}")