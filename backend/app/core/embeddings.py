# app/core/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

_model = None

def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        _model.max_seq_length = 256
    return _model


def encode_texts(texts, normalize=True):
    model = load_model()
    return model.encode(texts, normalize_embeddings=normalize)

def embed_and_pool(chunks):
    model = load_model()
    vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return vecs.mean(axis=0)

def chunk_by_tokens(text, tokenizer, max_wp=255):
    wp_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(wp_ids) <= max_wp:
        return [text]

    chunks = []
    for i in range(0, len(wp_ids), max_wp):
        chunk_ids = wp_ids[i: i + max_wp]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks

