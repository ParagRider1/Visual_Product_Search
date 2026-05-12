import pandas as pd

# Example experimental results
results = {
    "Configuration": [
        "CLIP Only",
        "CLIP + YOLO Crop",
        "CLIP + BLIP Fusion"
    ],

    "Recall@5": [
        0.60,
        0.70,
        0.80
    ],

    "NDCG@5": [
        0.52,
        0.63,
        0.71
    ],

    "mAP@5": [
        0.61,
        0.76,
        0.82
    ]
}

df = pd.DataFrame(results)

print("\nAblation Study Results")
print("-" * 50)

print(df)