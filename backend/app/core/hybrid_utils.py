# app/core/hybrid_utils.py

import pandas as pd
import numpy as np
from app.core.faiss_io import load_faiss_index
from app.core.embeddings import encode_texts
from app.core.config import settings
from sklearn.metrics.pairwise import cosine_similarity
from fastembed.embedding import SparseTextEmbedding
from transformers import CLIPProcessor, CLIPModel
import torch

# Load everything needed globally
metadata_df = pd.read_csv(settings.PROCESSED_CSV_PATH)
dense_index = load_faiss_index(settings.METADATA_INDEX_PATH)
image_index = load_faiss_index(settings.IMAGE_INDEX_PATH)
image_df = pd.read_csv("data/raw/image_embeddings_sorted.csv")
image_embeddings = np.array(image_df["image_embedding"].apply(eval).tolist(), dtype="float32")

splade_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Helpers
def sparse_to_dense(sparse_embedding, dim=30315):
    dense_vec = np.zeros(dim, dtype=np.float32)
    dense_vec[sparse_embedding.indices] = sparse_embedding.values
    return dense_vec

def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

sparse_vectors = np.vstack([
    sparse_to_dense(embedding, dim=30315)
    for embedding in metadata_df['splade_embedding']
]).astype('float32')

async def run_hybrid_search(query, top_k=5):
    # Dense
    dense_q = encode_texts([query], normalize=True)[0].astype('float32').reshape(1, -1)
    distances_dense, indices_dense = dense_index.search(dense_q, top_k)
    scores_dense = -distances_dense[0]

    # Sparse
    sparse_q = list(splade_model.embed([query]))[0]
    sparse_q_dense = sparse_to_dense(sparse_q, dim=30315).reshape(1, -1)
    similarities_sparse = cosine_similarity(sparse_q_dense, sparse_vectors)[0]
    top_indices_sparse = np.argpartition(-similarities_sparse, top_k)[:top_k]
    top_indices_sparse = top_indices_sparse[np.argsort(similarities_sparse[top_indices_sparse])[::-1]]
    scores_sparse = similarities_sparse[top_indices_sparse]

    # Image
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_image_embedding = clip_model.get_text_features(**inputs)
    query_image_embedding = query_image_embedding / query_image_embedding.norm(p=2, dim=-1, keepdim=True)
    query_image_embedding = query_image_embedding.cpu().numpy().astype('float32').reshape(1, -1)
    distances_image, indices_image = image_index.search(query_image_embedding, top_k)
    scores_image = -distances_image[0]

    # Normalize
    norm_dense = normalize_scores(scores_dense)
    norm_sparse = normalize_scores(scores_sparse)
    norm_image = normalize_scores(scores_image)

    # Hybrid Score
    hybrid_scores = 0.4 * norm_dense + 0.3 * norm_sparse + 0.3 * norm_image

    results = []
    for i in range(top_k):
        idx = indices_dense[0][i]
        row = metadata_df.iloc[idx]
        results.append({
            'Place Name': row['name'],
            'Neighborhood': row['neighborhood'],
            'Tags': row['tags'],
            'Description': row['short_description'],
            'Hybrid Score': hybrid_scores[i],
            'Dense Score': norm_dense[i],
            'Sparse Score': norm_sparse[i],
            'Image Score': norm_image[i]
        })

    results = [r for r in results if r['Hybrid Score'] > 0.1]
    results = sorted(results, key=lambda x: x['Hybrid Score'], reverse=True)

    return results
