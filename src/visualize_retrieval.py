from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

# Gallery images
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

# Retrieval results
results = []

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
        "path": gallery_path,
        "similarity": similarity.item()
    })

# Sort results
results = sorted(results, key=lambda x: x["similarity"], reverse=True)

# Visualization
fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))

# Query image
axes[0].imshow(query_image)
axes[0].set_title("Query")
axes[0].axis("off")

# Retrieved images
for i, result in enumerate(results):

    img = Image.open(result["path"]).convert("RGB")

    axes[i + 1].imshow(img)

    axes[i + 1].set_title(
        f"Rank {i+1}\n{result['similarity']:.3f}"
    )

    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()