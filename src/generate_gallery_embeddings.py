from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load CLIP
model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print("CLIP loaded successfully")

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Read evaluation file
eval_file = DATASET_ROOT / "Eval" / "list_eval_partition.txt"

with open(eval_file, "r") as f:
    lines = f.readlines()[2:]

data = []

for line in lines:
    parts = line.strip().split()

    data.append({
        "image_path": parts[0],
        "item_id": parts[1],
        "split": parts[2]
    })

df = pd.DataFrame(data)

# Use only gallery images
gallery_df = df[df["split"] == "gallery"]

# SMALL SUBSET for testing
gallery_df = gallery_df.head(100)

print(f"\nGallery images selected: {len(gallery_df)}")

# Storage
embeddings = []
image_paths = []
item_ids = []

# Generate embeddings
for _, row in tqdm(gallery_df.iterrows(), total=len(gallery_df)):

    img_path = DATASET_ROOT / "Img" / row["image_path"]

    try:
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            emb = model.get_image_features(**inputs)

        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output

        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

        embeddings.append(
            emb.cpu().numpy()[0]
        )

        image_paths.append(row["image_path"])
        item_ids.append(row["item_id"])

    except Exception as e:
        print(f"Error processing {img_path}")
        print(e)

# Convert to numpy
embeddings = np.array(embeddings)

print("\nEmbedding Matrix Shape:")
print(embeddings.shape)

# Save embeddings
Path("embeddings").mkdir(exist_ok=True)

np.save("embeddings/gallery_embeddings.npy", embeddings)

# Save metadata
metadata_df = pd.DataFrame({
    "image_path": image_paths,
    "item_id": item_ids
})

metadata_df.to_csv(
    "embeddings/gallery_metadata.csv",
    index=False
)

print("\nEmbeddings saved successfully")