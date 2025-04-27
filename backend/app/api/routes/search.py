# app/api/routes/search.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from app.core.faiss_io import load_faiss_index
from app.core.embeddings import encode_texts
from app.core.config import settings
import pandas as pd
import numpy as np
import math

router = APIRouter()

# Load FAISS index and metadata
faiss_index = load_faiss_index(settings.METADATA_INDEX_PATH)
metadata_df = pd.read_csv(settings.PROCESSED_CSV_PATH)

@router.get("/search")
async def search(q: str = Query(..., description="Search query"), top_k: int = 5):
    """
    Search places metadata based on text query.
    """
    try:
        query_vec = encode_texts([q], normalize=True)
        query_vec = np.array(query_vec).astype('float32')

        distances, indices = await run_in_threadpool(
            faiss_index.search,
            query_vec,
            top_k
        )

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # Invalid index
            if math.isnan(dist) or math.isinf(dist):
                continue  # Skip bad distance

            row = metadata_df.iloc[idx].to_dict()
            row['distance'] = max(float(dist), 0.0)  # Clamp to 0 if needed
            results.append(row)

        if not results:
            return JSONResponse(
                status_code=200,
                content={"results": [], "message": "No valid results found for your query."}
            )

        return {"results": results}
    
    except Exception as e:
        # Global error fallback
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {str(e)}"}
        )
