# ------------------------------
# Imports
# ------------------------------
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from transformers import CLIPProcessor

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
import os

# Define dataset paths
base_path = os.path.join(os.getcwd(), "data")
media_path = os.path.join(base_path, "media.csv")
places_path = os.path.join(base_path, "places.csv")
reviews_path = os.path.join(base_path, "reviews.csv")

# Load datasets with error handling
try:
    media_df = pd.read_csv(media_path)
    places_p2_df = pd.read_csv(places_path)
    reviews_df = pd.read_csv(reviews_path)
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check the dataset paths and ensure the files exist.")
    exit(1)

# Remove duplicates
reviews_df = reviews_df.drop_duplicates(subset=['place_id', 'review_text'])
media_df = media_df.drop_duplicates(subset=['place_id', 'media_url'])

# Aggregate reviews and media
reviews_agg = (
    reviews_df
    .groupby('place_id')['review_text']
    .apply(list)
    .reset_index()
)

media_agg = (
    media_df
    .groupby('place_id')['media_url']
    .apply(list)
    .reset_index()
)

# Merge datasets
merge_df = pd.merge(places_p2_df, media_agg, on='place_id', how='inner')
merge_df = pd.merge(merge_df, reviews_agg, on='place_id', how='inner')
merge_df = merge_df.drop_duplicates(subset='place_id').reset_index(drop=True)

# ------------------------------
# Helper Functions
# ------------------------------
def combining_text(row):
    """Combine text fields into a single string."""
    neighborhood = str(row['neighborhood']) if 'neighborhood' in row else ''
    tags = str(row['tags']) if 'tags' in row else ''
    short_description = str(row['short_description']) if 'short_description' in row else ''
    emojis = str(row['emojis']) if 'emojis' in row else ''
    return f"Neighborhood: {neighborhood}. Tags: {tags}. {short_description}. Emojis: {emojis}"

def concat_reviews(series):
    """Join all review texts in the group into a single string."""
    return ' '.join(series.astype(str))

def chunk_by_tokens(text, max_wp=510):
    """Split a long string into pieces whose WordPiece length ≤ max_wp."""
    wp_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(wp_ids) <= max_wp:
        return [text]

    chunks = []
    for i in range(0, len(wp_ids), max_wp):
        chunk_ids = wp_ids[i: i + max_wp]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks

def embed_and_pool(chunks):
    """
    Run the model on every chunk and return the arithmetic mean of the resulting vectors.
    """
    vecs = model_1.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return vecs.mean(axis=0)

def normalize(x):
    """Normalize a vector."""
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

# ------------------------------
# Text Embedding
# ------------------------------
# Combine text fields
merge_df['combined_text'] = merge_df.apply(combining_text, axis=1)

# Load model
model_1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = []
batch_size = 32
for i in tqdm(range(0, len(merge_df), batch_size), desc="Generating embeddings from dataset"):
    batch = merge_df['combined_text'].iloc[i: i + batch_size].tolist()
    batch_embeddings = model_1.encode(batch, normalize_embeddings=True)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)
merge_df['metadata_embedding'] = embeddings.tolist()

# Process reviews
agg_text_df = (
    reviews_df
    .groupby('place_id')['review_text']
    .apply(concat_reviews)
    .reset_index(name='all_reviews')
)

tokenizer = model_1.tokenizer
MAX_WP = 510

tqdm.pandas(desc="Chunking reviews")
agg_text_df['chunks'] = agg_text_df['all_reviews'].progress_apply(chunk_by_tokens)

tqdm.pandas(desc="Embedding and pooling")
agg_text_df['review_embedding'] = agg_text_df['chunks'].progress_apply(embed_and_pool)
agg_text_df['review_embedding'] = agg_text_df['review_embedding'].apply(lambda v: v.tolist())

# Merge review embeddings
merge_df = merge_df.merge(
    agg_text_df[['place_id', 'review_embedding']],
    how='left',
    left_on='place_id',
    right_on='place_id'
)

# ------------------------------
# FAISS Index Creation
# ------------------------------
# Metadata index
all_embeddings = np.array(merge_df['metadata_embedding'].tolist(), dtype='float32')
d = all_embeddings.shape[1]
index_metadata = faiss.IndexFlatL2(d)
index_metadata.add(all_embeddings)
print(f"Built FAISS metadata index with {index_metadata.ntotal} vectors of dimension {d}.")

# Review index
if 'review_embedding' in merge_df.columns:
    all_review_embeddings = np.array(merge_df['review_embedding'].tolist(), dtype='float32')
    index_reviews = faiss.IndexFlatL2(d)
    index_reviews.add(all_review_embeddings)
    print(f"Built FAISS review index with {index_reviews.ntotal} vectors of dimension {d}.")

# ------------------------------
# Search Functions
# ------------------------------
def search_places_metadata(query, index=index_metadata, top_k=5):
    query_embedding = model_1.encode([query], normalize_embeddings=True)[0].astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        row = merge_df.iloc[idx]
        results.append({
            'Place Name': row['name'],
            'Neighborhood': row['neighborhood'],
            'Tags': row['tags'],
            'Description': row['short_description'],
            'Distance': distances[0][i]
        })
    return results

def search_places_reviews(query, index=index_reviews, top_k=5):
    query_embedding = model_1.encode([query], normalize_embeddings=True)[0].astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        row = merge_df.iloc[idx]
        results.append({
            'Place Name': row['name'],
            'Neighborhood': row['neighborhood'],
            'Tags': row['tags'],
            'Description': row['short_description'],
            'Distance': distances[0][i]
        })
    return results

def search_places_image(query, index=index_metadata, top_k=5):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    query_embedding = model_1.get_text_features(**inputs)
    query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)
    query_embedding = query_embedding.detach().cpu().numpy().astype('float32').reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        row = merge_df.iloc[idx]
        results.append({
            'Place Name': row['name'],
            'Neighborhood': row['neighborhood'],
            'Tags': row['tags'],
            'Description': row['short_description'],
            'Distance': distances[0][i]
        })
    return results

# ------------------------------
# Main Execution
# ------------------------------

try:
    # Save the FAISS index
    faiss.write_index(index_metadata, "faiss_indices/index_file.index")
    print("FAISS metadata index has been successfully saved to 'index_file.index'.")
except Exception as e:
    print(f"An error occurred while saving the FAISS index: {e}")

merge_df.to_csv("data/merged.csv", index=False)
