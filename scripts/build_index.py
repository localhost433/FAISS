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
from app.core.embeddings import load_model, encode_texts, embed_and_pool, chunk_by_tokens
from app.core.faiss_io import build_faiss_index, save_faiss_index, METADATA_INDEX_PATH, REVIEWS_INDEX_PATH
from app.core.config import settings

# Paths
RAW_MEDIA = settings.RAW_MEDIA_PATH
RAW_PLACES = settings.RAW_PLACES_PATH
RAW_REVIEWS = settings.RAW_REVIEWS_PATH
PROCESSED_CSV = settings.PROCESSED_CSV_PATH
METADATA_INDEX_PATH = settings.METADATA_INDEX_PATH
REVIEWS_INDEX_PATH = settings.REVIEWS_INDEX_PATH

# Helpers
def parallel_apply(df, func, workers=4):
    """Apply a function in parallel to each row of the DataFrame."""
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(func, [row for _, row in df.iterrows()]))
    return results

def chunk_text_with_tokenizer(text):
    """Chunk text using the tokenizer from the loaded model."""
    model = load_model()
    tokenizer = model.tokenizer
    return chunk_by_tokens(text, tokenizer, max_wp=256)

def load_data():
    """Load datasets from CSV files."""
    print("Loading datasets.")
    media_df = pd.read_csv(RAW_MEDIA)
    places_df = pd.read_csv(RAW_PLACES)
    reviews_df = pd.read_csv(RAW_REVIEWS)
    return media_df, places_df, reviews_df

def prepare_merge_df(media_df, places_df, reviews_df):
    """Prepare the merged DataFrame for metadata and reviews."""
    print("Processing datasets.")
    reviews_df = reviews_df.drop_duplicates(subset=['place_id', 'review_text'])
    media_df = media_df.drop_duplicates(subset=['place_id', 'media_url'])

    reviews_agg = reviews_df.groupby('place_id')['review_text'].apply(list).reset_index()
    media_agg = media_df.groupby('place_id')['media_url'].apply(list).reset_index()

    merge_df = places_df.merge(media_agg, on='place_id', how='inner')
    merge_df = merge_df.merge(reviews_agg, on='place_id', how='inner')
    merge_df = merge_df.drop_duplicates(subset='place_id').reset_index(drop=True)
    return merge_df, reviews_df

def generate_metadata_embeddings(merge_df):
    """Generate metadata embeddings and build FAISS index."""
    print("Combining text fields in parallel.")
    merge_df['combined_text'] = parallel_apply(merge_df, combining_text, workers=os.cpu_count() // 2)

    print("Generating text embeddings.")
    model = load_model()
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(merge_df), batch_size), desc="Encoding combined texts"):
        batch = merge_df['combined_text'].iloc[i: i + batch_size].tolist()
        batch_embeddings = encode_texts(batch)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    merge_df['metadata_embedding'] = embeddings.tolist()
    return merge_df

def build_and_save_metadata_index(merge_df):
    """Build and save the FAISS index for metadata embeddings."""
    print("Building metadata FAISS index.")
    all_embeddings = np.array(merge_df['metadata_embedding'].tolist(), dtype='float32')
    metadata_index = build_faiss_index(all_embeddings)
    save_faiss_index(metadata_index, METADATA_INDEX_PATH)

def generate_review_embeddings(reviews_df):
    print("Combining and chunking reviews.")
    agg_text_df = reviews_df.groupby('place_id')['review_text'].apply(lambda s: ' '.join(s)).reset_index()

    model = load_model()
    tokenizer = model.tokenizer

    # Chunk all reviews
    with ProcessPoolExecutor(max_workers = min(8, os.cpu_count() // 2)) as executor:
        chunks_list = list(tqdm(executor.map(chunk_text_with_tokenizer, agg_text_df['review_text']),
                                total=len(agg_text_df), desc="Chunking reviews"))
    agg_text_df['chunks'] = chunks_list

    # Flatten chunks
    flat_chunks = []
    chunk_counts = []

    for chunks in agg_text_df['chunks']:
        flat_chunks.extend(chunks)
        chunk_counts.append(len(chunks))

    print(f"Total chunks to embed: {len(flat_chunks)}")

    # Embed all chunks in big batches
    batch_size = 128
    all_vecs = []

    for i in tqdm(range(0, len(flat_chunks), batch_size), desc="Encoding review chunks"):
        batch = flat_chunks[i:i + batch_size]
        batch_vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(batch_vecs)

    all_vecs = np.vstack(all_vecs)  # (N_chunks, dim)

    # Pool efficiently
    pooled_vecs = []

    idx = 0
    for count in chunk_counts:
        if count == 0:
            pooled_vec = np.zeros(all_vecs.shape[1])  # Empty, fallback
        else:
            pooled_vec = all_vecs[idx:idx + count].mean(axis=0)
        pooled_vecs.append(pooled_vec)
        idx += count

    agg_text_df['review_embedding'] = [v.tolist() for v in pooled_vecs]

    return agg_text_df

def build_and_save_review_index(merge_df):
    """Build and save the FAISS index for review embeddings."""
    if 'review_embedding' in merge_df.columns:
        print("Building review FAISS index.")
        all_review_embeddings = np.array(merge_df['review_embedding'].tolist(), dtype='float32')
        review_index = build_faiss_index(all_review_embeddings)
        save_faiss_index(review_index, REVIEWS_INDEX_PATH)

def save_processed_csv(merge_df):
    """Save the merged DataFrame to a CSV file."""
    print(f"Saving merged CSV at {PROCESSED_CSV}...")
    PROCESSED_CSV.parent.mkdir(exist_ok=True, parents=True)
    merge_df.to_csv(PROCESSED_CSV, index=False)

def main():
    # Load datasets
    media_df, places_df, reviews_df = load_data()
    merge_df, reviews_df = prepare_merge_df(media_df, places_df, reviews_df)

    # Generate metadata embeddings and build FAISS index
    merge_df = generate_metadata_embeddings(merge_df)
    build_and_save_metadata_index(merge_df)

    # Generate review embeddings and build FAISS index
    agg_text_df = generate_review_embeddings(reviews_df)
    merge_df = merge_df.merge(agg_text_df[['place_id', 'review_embedding']], on='place_id', how='left')

    # Build and save review index
    build_and_save_review_index(merge_df)

    # Save processed CSV
    save_processed_csv(merge_df)

    print("Index building completed successfully.")

if __name__ == "__main__":
    main()
