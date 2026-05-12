import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset root
DATASET_ROOT = Path("data/DeepFashion")

# Files
BBOX_FILE = DATASET_ROOT / "Anno" / "list_bbox_inshop.txt"

# Read bbox file
with open(BBOX_FILE, "r") as f:
    lines = f.readlines()

# Remove header lines
lines = lines[2:]

bbox_data = []

for line in lines:
    parts = line.strip().split()

    image_path = parts[0]

    x1 = int(parts[-4])
    y1 = int(parts[-3])
    x2 = int(parts[-2])
    y2 = int(parts[-1])

    bbox_data.append({
        "image_path": image_path,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    })

# Create dataframe
bbox_df = pd.DataFrame(bbox_data)

# Select one sample
sample = bbox_df.iloc[0]

# Full image path
img_path = DATASET_ROOT / "Img" / sample["image_path"]

# Read image
image = cv2.imread(str(img_path))

# Convert BGR -> RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw rectangle
cv2.rectangle(
    image,
    (sample["x1"], sample["y1"]),
    (sample["x2"], sample["y2"]),
    (255, 0, 0),
    3
)

# Show image
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.title("Bounding Box Visualization")
plt.axis("off")

plt.show()