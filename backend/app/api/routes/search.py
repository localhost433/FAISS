# app/api/routes/search.py

from fastapi import APIRouter, Query
from app.core.faiss_io import load_faiss_index
from app.core.embeddings import encode_texts
from app.core.config import settings
import pandas as pd
import numpy as np


router = APIRouter()

# Load FAISS index and metadata
faiss_index = load_faiss_index(settings.METADATA_INDEX_PATH)
metadata_df = pd.read_csv(settings.PROCESSED_CSV_PATH)

@router.get("/search")
async def search(q: str = Query(..., description="Search query"), top_k: int = 5):
    """
    Search places metadata based on text query.
    """
    query_vec = encode_texts([q], normalize=True)
    query_vec = np.array(query_vec).astype('float32')

    distances, indices = faiss_index.search(query_vec, top_k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # Skip invalid
        row = metadata_df.iloc[idx].to_dict()
        row['distance'] = float(dist)
        results.append(row)

    return {"results": results}
