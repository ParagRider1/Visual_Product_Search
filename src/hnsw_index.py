import hnswlib
import numpy as np
import pandas as pd

# Load embeddings
embeddings = np.load(
    "embeddings/gallery_embeddings.npy"
)

# Load metadata
metadata = pd.read_csv(
    "embeddings/gallery_metadata.csv"
)

print("Embeddings shape:")
print(embeddings.shape)

# Embedding dimension
dim = embeddings.shape[1]

# Create HNSW index
index = hnswlib.Index(
    space='cosine',
    dim=dim
)

# Initialize index
index.init_index(
    max_elements=len(embeddings),
    ef_construction=200,
    M=16
)

# Add embeddings
index.add_items(
    embeddings,
    np.arange(len(embeddings))
)

# Set search parameter
index.set_ef(50)

print("\nHNSW index built successfully")

# Use first embedding as query
query_vector = embeddings[0]

# Search top-5 nearest neighbors
labels, distances = index.knn_query(
    query_vector,
    k=5
)

print("\nTop Retrieval Results")
print("-" * 40)

for rank, (label, distance) in enumerate(
    zip(labels[0], distances[0]),
    start=1
):

    similarity = 1 - distance

    image_path = metadata.iloc[label]["image_path"]

    print(
        f"{rank}. {image_path} --> Similarity: {similarity:.4f}"
    )

# Save index
index.save_index("indexes/fashion_hnsw.index")

print("\nIndex saved successfully")