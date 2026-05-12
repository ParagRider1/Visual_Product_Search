import pandas as pd
from pathlib import Path

# Dataset paths
DATASET_ROOT = Path("data/DeepFashion")

EVAL_FILE = DATASET_ROOT / "Eval" / "list_eval_partition.txt"

# Read file
with open(EVAL_FILE, "r") as f:
    lines = f.readlines()

# Remove first two header lines
lines = lines[2:]

data = []

for line in lines:
    parts = line.strip().split()

    image_path = parts[0]
    item_id = parts[1]
    split = parts[2]

    data.append({
        "image_path": image_path,
        "item_id": item_id,
        "split": split
    })

# Create dataframe
df = pd.DataFrame(data)

# Show basic information
print("\nDataset Overview")
print("-" * 50)

print(f"Total images: {len(df)}")

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nUnique item IDs:")
print(df["item_id"].nunique())

print("\nFirst 5 rows:")
print(df.head())