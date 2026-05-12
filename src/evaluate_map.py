from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

# ----------------------------
# Load CLIP
# ----------------------------

model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print("CLIP loaded successfully")

# ----------------------------
# Dataset root
# ----------------------------

DATASET_ROOT = Path("data/DeepFashion")

# ----------------------------
# Read evaluation file
# ----------------------------

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

# ----------------------------
# Small subset
# ----------------------------

query_df = df[df["split"] == "query"].head(10)
gallery_df = df[df["split"] == "gallery"].head(100)

print(f"\nQueries: {len(query_df)}")
print(f"Gallery: {len(gallery_df)}")

# ----------------------------
# Gallery embeddings
# ----------------------------

gallery_embeddings = []

for _, row in gallery_df.iterrows():

    img_path = DATASET_ROOT / "Img" / row["image_path"]

    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    if hasattr(emb, "pooler_output"):
        emb = emb.pooler_output

    emb = F.normalize(emb, p=2, dim=-1)

    gallery_embeddings.append(
        emb.cpu().numpy()[0]
    )

gallery_embeddings = np.array(gallery_embeddings)

# ----------------------------
# mAP@5
# ----------------------------

k = 5
ap_scores = []

for _, query_row in query_df.iterrows():

    query_path = DATASET_ROOT / "Img" / query_row["image_path"]

    query_image = Image.open(query_path).convert("RGB")

    query_inputs = processor(
        images=query_image,
        return_tensors="pt"
    )

    with torch.no_grad():
        query_emb = model.get_image_features(
            **query_inputs
        )

    if hasattr(query_emb, "pooler_output"):
        query_emb = query_emb.pooler_output

    query_emb = F.normalize(
        query_emb,
        p=2,
        dim=-1
    )

    query_emb = query_emb.cpu().numpy()[0]

    # Similarity scores
    similarities = np.dot(
        gallery_embeddings,
        query_emb
    )

    # Relevance labels
    relevance = (
        gallery_df["item_id"]
        == query_row["item_id"]
    ).astype(int).values

    # Top-K sorting
    sorted_idx = np.argsort(
        similarities
    )[::-1][:k]

    topk_scores = similarities[sorted_idx]
    topk_relevance = relevance[sorted_idx]

    # Skip if no positives
    if np.sum(topk_relevance) == 0:
        continue

    ap = average_precision_score(
        topk_relevance,
        topk_scores
    )

    ap_scores.append(ap)

# Final mAP
mean_ap = np.mean(ap_scores)

print(f"\nmAP@{k}: {mean_ap:.4f}")