# scripts/build_index.py

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.text_utils import combining_text, concat_reviews
from app.core.embeddings import load_model, encode_texts, chunk_by_tokens
from app.core.faiss_io import build_faiss_index, save_faiss_index, METADATA_INDEX_PATH, IMAGE_INDEX_PATH
from app.core.config import settings
from fastembed.embedding import SparseTextEmbedding

# Paths
RAW_MEDIA = settings.RAW_MEDIA_PATH
RAW_PLACES = settings.RAW_PLACES_PATH
RAW_REVIEWS = settings.RAW_REVIEWS_PATH
PROCESSED_CSV = settings.PROCESSED_CSV_PATH

# Helpers
def parallel_apply(df, func, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(func, [row for _, row in df.iterrows()]))
    return results

def load_data():
    print("Loading datasets.")
    media_df = pd.read_csv(RAW_MEDIA)
    places_df = pd.read_csv(RAW_PLACES)
    reviews_df = pd.read_csv(RAW_REVIEWS)
    return media_df, places_df, reviews_df

def prepare_merge_df(media_df, places_df, reviews_df):
    print("Processing datasets.")
    reviews_df = reviews_df.drop_duplicates(subset=['place_id', 'review_text'])
    media_df = media_df.drop_duplicates(subset=['place_id', 'media_url'])

    reviews_agg = reviews_df.groupby('place_id')['review_text'].apply(lambda x: ' '.join(x)).reset_index(name='reviews')
    media_agg = media_df.groupby('place_id')['media_url'].apply(list).reset_index()

    merge_df = places_df.merge(media_agg, on='place_id', how='inner')
    merge_df = merge_df.merge(reviews_agg, on='place_id', how='inner')
    merge_df = merge_df.drop_duplicates(subset='place_id').reset_index(drop=True)
    return merge_df

def combine_all_text(row):
    return (
        f"Name: {row.get('name', '')}. "
        f"Neighborhood: {row.get('neighborhood', '')}. "
        f"Tags: {row.get('tags', '')}. "
        f"Description: {row.get('short_description', '')}. "
        f"Emojis: {row.get('emojis', '')}. "
        f"User Reviews: {row.get('reviews', '')}."
    )

def generate_hybrid_embeddings(merge_df):
    print("Combining metadata and reviews for hybrid embeddings.")
    merge_df['combined_text'] = parallel_apply(merge_df, combine_all_text, workers=os.cpu_count() // 2)

    print("Generating dense text embeddings.")
    model = load_model()
    batch_size = 64
    dense_embeddings = []
    for i in tqdm(range(0, len(merge_df), batch_size), desc="Encoding dense texts"):
        batch = merge_df['combined_text'].iloc[i:i+batch_size].tolist()
        encoded = model.encode(batch, normalize_embeddings=True)
        dense_embeddings.append(encoded)
    dense_embeddings = np.vstack(dense_embeddings)
    merge_df['metadata_embedding'] = dense_embeddings.tolist()

    print("Generating SPLADE sparse embeddings.")
    splade_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    splade_embeddings = []
    for i in tqdm(range(0, len(merge_df), 64), desc="Embedding with SPLADE"):
        batch = merge_df['combined_text'].iloc[i:i+64].tolist()
        splade_embeddings.extend(splade_model.embed(batch))
    merge_df['splade_embedding'] = splade_embeddings

    return merge_df

def save_processed_csv(merge_df):
    print(f"Saving merged CSV at {PROCESSED_CSV}...")
    PROCESSED_CSV.parent.mkdir(exist_ok=True, parents=True)
    merge_df.to_csv(PROCESSED_CSV, index=False)

def main():
    media_df, places_df, reviews_df = load_data()
    merge_df = prepare_merge_df(media_df, places_df, reviews_df)

    merge_df = generate_hybrid_embeddings(merge_df)
    dense_index = build_faiss_index(np.array(merge_df['metadata_embedding'].tolist(), dtype='float32'))
    save_faiss_index(dense_index, METADATA_INDEX_PATH)

    save_processed_csv(merge_df)

    try:
        print("Loading image embeddings for FAISS index...")
        image_df = pd.read_csv("data/raw/image_embeddings_sorted.csv")
        image_df = image_df.dropna(subset=["image_embedding"])
        image_embeddings = image_df["image_embedding"].apply(eval).tolist()
        image_embeddings = np.array(image_embeddings, dtype="float32")
        image_index = build_faiss_index(image_embeddings)
        save_faiss_index(image_index, IMAGE_INDEX_PATH)
        print("Image FAISS index saved.")
    except Exception as e:
        print("Skipping image embedding index:", e)

    print("Index building completed successfully.")

if __name__ == "__main__":
    main()
