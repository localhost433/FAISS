# scripts/build_index.py

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os
import sys
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

# Now safe to import
from app.core.text_utils import combining_text, concat_reviews
from app.core.embeddings import load_model, encode_texts, embed_and_pool, chunk_by_tokens
from app.core.faiss_io import build_faiss_index, save_faiss_index, METADATA_INDEX_PATH, REVIEWS_INDEX_PATH
from app.core.config import settings

# Use paths from settings
RAW_MEDIA = settings.RAW_MEDIA_PATH
RAW_PLACES = settings.RAW_PLACES_PATH
RAW_REVIEWS = settings.RAW_REVIEWS_PATH
PROCESSED_CSV = settings.PROCESSED_CSV_PATH
METADATA_INDEX_PATH = settings.METADATA_INDEX_PATH
REVIEWS_INDEX_PATH = settings.REVIEWS_INDEX_PATH

def main():
    print("Loading datasets...")
    media_df = pd.read_csv(RAW_MEDIA)
    places_df = pd.read_csv(RAW_PLACES)
    reviews_df = pd.read_csv(RAW_REVIEWS)

    # Preprocess
    print("Processing datasets...")
    reviews_df = reviews_df.drop_duplicates(subset=['place_id', 'review_text'])
    media_df = media_df.drop_duplicates(subset=['place_id', 'media_url'])

    reviews_agg = reviews_df.groupby('place_id')['review_text'].apply(list).reset_index()
    media_agg = media_df.groupby('place_id')['media_url'].apply(list).reset_index()

    merge_df = places_df.merge(media_agg, on='place_id', how='inner')
    merge_df = merge_df.merge(reviews_agg, on='place_id', how='inner')
    merge_df = merge_df.drop_duplicates(subset='place_id').reset_index(drop=True)

    # Text combination
    merge_df['combined_text'] = merge_df.apply(combining_text, axis=1)

    # Generate embeddings
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(merge_df), batch_size), desc="Generating embeddings from dataset"):
        batch = merge_df['combined_text'].iloc[i: i + batch_size].tolist()
        batch_embeddings = encode_texts(batch)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    merge_df['metadata_embedding'] = embeddings.tolist()

    # Build and save index for metadata
    all_embeddings = np.array(merge_df['metadata_embedding'].tolist(), dtype='float32')
    metadata_index = build_faiss_index(all_embeddings)
    save_faiss_index(metadata_index, METADATA_INDEX_PATH)


    # Review text processing
    tokenizer = load_model().tokenizer
    max_token_limit = 256  # Set the maximum token limit for the model

    # Combine all reviews into a single text per place
    agg_text_df = reviews_df.groupby('place_id')['review_text'].apply(lambda s: ' '.join(s)).reset_index()

    # Split long texts into chunks
    print("Chunking long review texts...")
    agg_text_df['chunks'] = agg_text_df['review_text'].progress_apply(lambda text: chunk_by_tokens(text, tokenizer, max_token_limit))

    # Embed and pool the chunks
    print("Generating review embeddings...")
    agg_text_df['review_embedding'] = agg_text_df['chunks'].progress_apply(embed_and_pool)
    agg_text_df['review_embedding'] = agg_text_df['review_embedding'].apply(lambda v: v.tolist())

    # Merge review embeddings
    merge_df = merge_df.merge(agg_text_df[['place_id', 'review_embedding']], on='place_id', how='left')

    # Embedding review text
    if 'review_embedding' in merge_df.columns:
        print("Generating review embeddings...")
        all_review_embeddings = np.array(merge_df['review_embedding'].tolist(), dtype='float32')
        review_index = build_faiss_index(all_review_embeddings)
        save_faiss_index(review_index, REVIEWS_INDEX_PATH)

    # Save merged csv
    PROCESSED_CSV.parent.mkdir(exist_ok=True, parents=True)
    merge_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Merged CSV saved at {PROCESSED_CSV}")

if __name__ == "__main__":
    main()
