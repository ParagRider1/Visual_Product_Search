from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration
)

from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F

# ----------------------------
# Load CLIP
# ----------------------------

clip_model_name = "openai/clip-vit-base-patch32"

print("Loading CLIP...")

clip_model = CLIPModel.from_pretrained(
    clip_model_name
)

clip_processor = CLIPProcessor.from_pretrained(
    clip_model_name
)

print("CLIP loaded")

# ----------------------------
# Load BLIP
# ----------------------------

blip_model_name = "Salesforce/blip-image-captioning-base"

print("Loading BLIP...")

blip_processor = BlipProcessor.from_pretrained(
    blip_model_name
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    blip_model_name
)

print("BLIP loaded")

# ----------------------------
# Dataset image
# ----------------------------

DATASET_ROOT = Path("data/DeepFashion")

img_path = DATASET_ROOT / "Img" / "img" / "WOMEN" / "Dresses" / "id_00000002" / "02_1_front.jpg"

image = Image.open(img_path).convert("RGB")

# ----------------------------
# Generate BLIP caption
# ----------------------------

blip_inputs = blip_processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    blip_output = blip_model.generate(
        **blip_inputs
    )

caption = blip_processor.decode(
    blip_output[0],
    skip_special_tokens=True
)

print("\nGenerated Caption:")
print(caption)

# ----------------------------
# CLIP Image Embedding
# ----------------------------

image_inputs = clip_processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    image_emb = clip_model.get_image_features(
        **image_inputs
    )

if hasattr(image_emb, "pooler_output"):
    image_emb = image_emb.pooler_output

image_emb = F.normalize(image_emb, p=2, dim=-1)

# ----------------------------
# CLIP Text Embedding
# ----------------------------

text_inputs = clip_processor(
    text=[caption],
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    text_emb = clip_model.get_text_features(
        **text_inputs
    )

if hasattr(text_emb, "pooler_output"):
    text_emb = text_emb.pooler_output

text_emb = F.normalize(text_emb, p=2, dim=-1)

# ----------------------------
# Fusion
# ----------------------------

alpha = 0.7

fusion_emb = (
    alpha * image_emb
    + (1 - alpha) * text_emb
)

fusion_emb = F.normalize(
    fusion_emb,
    p=2,
    dim=-1
)

print("\nFusion Embedding Shape:")
print(fusion_emb.shape)

print("\nFirst 10 Values:")
print(fusion_emb[0][:10])